# KNN

### Installation

To setup a virtual environment with everything needed to run this project, you can run the script `setup_env.sh`. If there are some problems with the setup process, this is the installaiton guide from the original InterObject3D article this is based on:

Preferably install a python virtual env (conda) using the requirements file in the repository or use it as a guideline since the ME engine needs to be installed seperately.
The code is based on the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine), and the [documentation page](https://nvidia.github.io/MinkowskiEngine/overview.html#installation) contains useful instructions on how to install the Minkowski Engine.

## Training

Run training by running:
```bash
python train.py
```
**Training Parameters:**
* `--batch_size`: Batch size
* `--max_epochs`: Max number of epochs
* `--lr`: Learning rate
* `--momentum`: Momentum
* `--weight_decay`: Weight decay
* `--weights`: Path to weights, if training pre-trained model
* `--save_weights`: Name of saved weights
* `--backbone`: The backbone network used for training: `pointnet` for pointnet, `hpointnet` for full hierarchic pointnet, `"hpointnetsmall` for small hierarchic pointnet and anything else for default MinkUNet34C
* `--dataset`: Path to the raw dataset

If there is no processed dataset in `data_preparation/processed_datasets`, the raw dataset on the path `--dataset` is processed there first before training.

## Start Evaluation

**Note:** The starting directory is considered to be `Minkowski/training`.

The `run_stat.py` script generates modified files from the SemanticKITTI datasets. For this to work, you need to place the raw dataset into the corresponding directory:
* `data_preparation/dataset/kitti/`

You can then generate these files by running the following commands:

```bash
# For the KITTI dataset
python run_stat.py --generate_scenes_kitti

``` 

This will create the `processed_datasets` directory. For demonstration purposes, one scene is already generated in the `processed_datasets` directory.

### Running Evaluation on Processed Datasets

Before starting the evaluation, create the `dataset_result/kitti/` directory to store the `.csv` evaluation results.

**Evaluation Parameters:**
You can set these parameters directly in the `run_stat.py` file or add them during command execution:
* `--pretraining_weights`: Path to the model weights to be used.
* `--used_model`: Type of model architecture.
* `--dataset`: The dataset to be evaluated (`kitti`).
* `--seq_num`: (KITTI only) The specific sequence directory to use for data.

Start the evaluation by running:
```bash
python run_stat.py --run_interaction
```

This will generate the result `.csv` files. To calculate the final statistics from these files, run:
```bash
python run_stat.py --run_stats
```

## Start interactive segmentation part 

You can run the segmentation tool in an interactive mode where a user provides clicks manually to guide the network. This is done using the `run_inter3d.py` script. 

To enable this mode, use the `--real_user=True` flag and specify a single object instance to segment using `--no-all_instances` and `--instance_counter_id`.

**Key Parameters:**
* `--real_user=True`: Launches the interactive visualizer allowing you to manually click to provide positive/negative feedback.
* `--verbal=True`: Enables detailed console output --  current IoU, number of clicks.
* `--no-all_instances`: Prevents the script from looping through the entire dataset, restricting it to a single target.
* `--instance_counter_id`: The specific ID of the object instance you want to segment.
* `--pretraining_weights` & `--used_model`: Path to the model weights and the architecture name.
* `--dataset_scenes`: Path to the specific scene or frame to be loaded.

### Example Commands

**For the SemanticKITTI dataset:**
```bash
python3 run_inter3d.py \
  --real_user=True \
  --verbal=True \
  --dataset=kitti \
  --no-all_instances \
  --instance_counter_id=1 \
  --pretraining_weights=weights/weights_exp14_3_pointN.pth \
  --used_model=HierarchicPointNetSeg \
  --dataset_scenes=data_preparation/processed_datasets/seq00/frame000000
  ``` 