import os
import glob
import subprocess
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from data_preparation.kitti import preprocess_data


def generate_scenes_scannet():
    PATH_TO_SCANNET_DATASET = 'data_preparation/dataset/scannet'
    OUTPUT_DIR = 'data_preparation/processed_datasets_scannet/'

    for folder_name in os.listdir(PATH_TO_SCANNET_DATASET):
        print(f"Executing script for: {folder_name}")

        command = [
                "python", "data_preparation/main_scannet.py",
                f"--name={folder_name}",
                f"--path={PATH_TO_SCANNET_DATASET}",
                f"--output_dir={OUTPUT_DIR}"
            ]
        
        subprocess.run(command, check=True)

    print("All directories completed.")

def generate_scenes_kitti():
    PATH_TO_KITTI_DATASET = 'data_preparation/dataset/kitti'
    preprocess_data(PATH_TO_KITTI_DATASET)


def run_inter_scannet():
    
    path_to_scenes = "data_preparation/processed_datasets_scannet/scenes_&_classes/"
    masks = "data_preparation/processed_datasets_scannet/masks5x5/"
    crops = "data_preparation/processed_datasets_scannet/crops5x5/"
    pretraining_weights = config.pretraining_weights
    used_model = config.used_model
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
                f"--used_model={used_model}",
                f"--pretraining_weights={pretraining_weights}",
                f"--dataset={dataset}",
                "--save_results_file=True",
                f"--results_path=./dataset_result/{dataset}/",
                f"--results_file_name={results_file_name}"
            ]

        # check - stop if run_inter3d.py ends with error
        subprocess.run(command, check=True)

    print("All directories completed.")

def run_inter_semkitti():
    seq_num = config.seq_num
    path_to_scenes = f"data_preparation/processed_datasets/seq{seq_num}"
    pretraining_weights = config.pretraining_weights
    used_model = config.used_model
    dataset = config.dataset

    for folder_name in sorted(os.listdir(path_to_scenes)):
        folder_path = os.path.join(path_to_scenes, folder_name)

        if not os.path.isdir(folder_path):
            continue

        results_file_name = folder_name
        print(f"Executing script for: {results_file_name}")

        command = [
                "python", "run_inter3d.py",
                f"--dataset_scenes={folder_path}",  # Pass the whole frame folder
                "--cubeedge=0.05",
                f"--used_model={used_model}",
                f"--pretraining_weights={pretraining_weights}",
                f"--dataset={dataset}",
                "--save_results_file=True",
                f"--results_path=./dataset_result/{dataset}/",
                f"--results_file_name={results_file_name}"
            ]

        # check - stop if run_inter3d.py ends with error
        subprocess.run(command, check=True)

    print("All directories completed.")

def run_stat():

    result_file_paths = glob.glob(f"dataset_result/{config.dataset}/*.csv")
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

    parser.add_argument('--generate_scenes_scannet', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--generate_scenes_kitti', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--run_interaction', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--run_stats', action=argparse.BooleanOptionalAction, default=False)

    ###############
    parser.add_argument('--pretraining_weights', type=str, default='weights/weights_exp14_11_pointnet.pth')
    parser.add_argument('--used_model', type=str, default='MinkowskiPointNetSeg') # MinkUNet34C, MinkUNet18B, MinkowskiPointNetSeg
    parser.add_argument('--dataset', type=str, default='kitti') # scannet, kitti
    ##############

    ## sequence num for kitti dataset, used with --run_interaction
    parser.add_argument('--seq_num', type=str, default='00')


    config = parser.parse_args()

    if config.generate_scenes_scannet:
        generate_scenes_scannet()

    if config.generate_scenes_kitti:
        generate_scenes_kitti()

    if config.run_interaction:
        if config.dataset == 'scannet':
            run_inter_scannet()
        elif config.dataset == 'kitti':
            run_inter_semkitti()

    if config.run_stats:
        run_stat()


# python run_stat.py --run_stats
# python run_stat.py --run_interaction

# python run_stat.py --generate_scenes_scannet