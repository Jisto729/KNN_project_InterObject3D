import os
import glob
import subprocess
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score



def generate_scenes_scannet():
    PATH_TO_SCANNET_DATASET = '../../datasetgen/example'
    OUTPUT_DIR = '../../datasetgen/results'

    for folder_name in os.listdir(PATH_TO_SCANNET_DATASET):
        print(f"Executing script for: {folder_name}")

        command = [
                "python", "../../datasetgen/main_scannet.py",
                f"--name={folder_name}",
                f"--path={PATH_TO_SCANNET_DATASET}",
                f"--output_dir={OUTPUT_DIR}"
            ]
        
        subprocess.run(command, check=True)

    print("All directories completed.")


def run_inter():
    path_to_scenes = config.path_to_scene
    masks = config.path_to_mask
    crops = config.path_to_cops
    model_used = config.pretraining_weights
    dataset = config.dataset

    for folder_name in os.listdir(path_to_scenes):
        folder_path = os.path.join(path_to_scenes, folder_name)

        if os.path.isdir(folder_path):

            results_file_name = folder_name
            
            txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
            npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

            if not txt_files or not npy_files:
                print(f"Skipping directory {folder_name}: Missing .txt or .npy file.")
                continue

            classes = txt_files[0]
            scenes = npy_files[0]

        print(f"Executing script for: {results_file_name}")

        command = [
                "python", "run_inter3d.py",
                f"--dataset_classes={classes}",
                f"--dataset_scenes={scenes}",
                f"--dataset_folder_scene={crops}",
                f"--dataset_folder_masks={masks}",
                "--cubeedge=0.05",
                f"--dataset={dataset}",
                f"--pretraining_weights={model_used}",
                "--dataset=scannet",
                "--save_results_file=True",
                f"--results_file_name={results_file_name}"
            ]

        # check - stop if run_inter3d.py ends with error
        subprocess.run(command, check=True)

    print("All directories completed.")


def run_stat():

    result_file_paths = glob.glob(f"dataset_mini/results/{config.dataset_result}/*.csv")
    df_list = [pd.read_csv(f, sep='\s+', header=None, 
                        names=['index', 'scene', 'obj_id', 'click', 'iou', 'confidence']) 
            for f in result_file_paths]
    df = pd.concat(df_list, ignore_index=True)


    target_clicks = [1, 2, 3, 5, 10, 20]
  

    results = []



    for click in target_clicks:
        click_data = df[df['click'] == click]

        click_data = click_data[click_data != None]

        if len(click_data) == 0:
            continue
        
        iou_scores = click_data['iou']
        conf_scores = click_data['confidence']


        miou = iou_scores.mean()

        sr25 = (iou_scores > 25.0).mean() * 100

        sr50 = (iou_scores > 50.0).mean() * 100

        thresholds = np.arange(50,100,5)
        msr = np.mean([(iou_scores >= t).mean() * 100 for t in thresholds])

        # AP@25%
        y_true_25 = (iou_scores >= 25.0).astype(int)
        ap25 = average_precision_score(y_true_25, conf_scores) * 100 if y_true_25.sum() > 0 else 0.0


        # AP@50%
        y_true_50 = (iou_scores >= 50.0).astype(int)
        ap50 = average_precision_score(y_true_50, conf_scores) * 100 if y_true_50.sum() > 0 else 0.0

        # MEAN AP [50-100 %]
        ap_scores = []
        for t in thresholds:
            y_true_t = (iou_scores >= t).astype(int)
            if y_true_t.sum() > 0:
                ap_t = average_precision_score(y_true_t, conf_scores)
            else:
                ap_t = 0.0
            ap_scores.append(ap_t)
            
        map_score = np.mean(ap_scores) * 100


        results.append({
            'Method': f'Ours ({click} clicks)',
            'mSR': f"{msr:.1f}",
            'SR@50': f"{sr50:.1f}",
            'SR@25': f"{sr25:.1f}",
            'mAP': f"{map_score:.1f}",
            'AP@50': f"{ap50:.1f}",
            'AP@25': f"{ap25:.1f}",
            "mIoU" : f"{miou:.2f}"
        })
        
    results_df = pd.DataFrame(results)

    print(results_df.to_markdown(index=False))

    model_name = os.path.splitext(os.path.basename(config.pretraining_weights))[0]
    save_path = f'results/{model_name}.md'

    with open(save_path, 'w') as f:
        f.write(results_df.to_markdown(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_scenes_scannet', type=bool, default=False)

    parser.add_argument('--run_interaction', type=bool, default=False)
    parser.add_argument('--path_to_scene', type=str, default='../../datasetgen/resultsscenes_&_classes/')
    parser.add_argument('--path_to_mask', type=str, default='../../datasetgen/resultsmasks5x5/')
    parser.add_argument('--path_to_cops', type=str, default='../../datasetgen/resultscrops5x5/')
    parser.add_argument('--pretraining_weights', type=str, default='weights/weights_exp14_14_default.pth')
    parser.add_argument('--dataset', type=str, default='scannet')

    parser.add_argument('--run_stats', type=bool, default=False)
    parser.add_argument('--dataset_result', type=str, default='kitti')

    PATH_TO_SCENES_CLASSES = '../../datasetgen/resultsscenes_&_classes/'
    MASKS = '../../datasetgen/resultsmasks5x5/'
    CROPS = '../../datasetgen/resultscrops5x5/'

    config = parser.parse_args()

    if config.generate_scenes_scannet:
        generate_scenes_scannet()

    if config.run_interaction:
        run_inter()

    if config.run_stats:
        run_stat()
# python run_stat.py --run_stats=True 