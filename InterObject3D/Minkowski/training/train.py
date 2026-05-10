import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.pointnet import (
    MinkowskiPointNetSeg,
    HierarchicPointNetSeg,
    SmallHierarchicPointNetSeg,
)
from torch.utils.tensorboard import SummaryWriter
import data_preparation.kitti

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Had some issues with open3d
os.environ["OPEN3D_CPU_RENDERING"] = "true"
write_path = "/mnt/d/downloads/KNN/weights/"


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Binary mask generation
    if config.backbone == "pointnet":
        print("pointnet")
        net = MinkowskiPointNetSeg(in_channel=2, out_channel=2, dimension=3).to(device)
    elif config.backbone == "hpointnet":
        print("Hierarchic pointnet")
        net = HierarchicPointNetSeg(in_channel=2, out_channel=2, dimension=3).to(device)
    elif config.backbone == "hpointnetsmall":
        print("Small hierarchic pointnet")
        net = SmallHierarchicPointNetSeg(in_channel=2, out_channel=2, dimension=3).to(
            device
        )
    else:
        net = MinkUNet34C(in_channels=2, out_channels=2, D=3).to(device)

    net = net.to(device)
    # If use pre-training weights
    if os.path.exists(config.weights):
        model_dict = torch.load(config.weights)
        net.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Dataset, data loader
    data_preparation.kitti.preprocess_data(
        "/mnt/d/downloads/KNN/data_odometry_velodyne/dataset"
    )
    train_dataset = data_preparation.kitti.KittiDataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        # 1) collate_fn=collation_fn,
        # 2) collate_fn=ME.utils.batch_sparse_collate,
        # 3) collate_fn=ME.utils.SparseCollation(),
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=1,
    )

    accum_loss, accum_iter, tot_iter = 0, 0, 0
    intersection, mintersection, union = 0, 0, 0

    writer = SummaryWriter(config.save_writer)
    net.train()
    grad_dict_total = {k: 0 for k, v in net.named_parameters()}
    N_samples = 0

    for epoch in range(config.max_epochs):
        train_iter = iter(train_dataloader)

        # Training
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data

            input = ME.SparseTensor(feats.float(), coords, device=device)
            out = net(input)
            try:
                optimizer.zero_grad()
                loss = criterion(out.F.squeeze(), labels.long().to(device))
                loss.backward(retain_graph=True)
                optimizer.step()
            except (RuntimeError, MemoryError) as error:  # Handle OOM
                print(f"+++ Recovering from exception: {error} +++")
                torch.cuda.empty_cache()
                continue
            optimizer.zero_grad()
            outputs = net(input)
            l2_norm = torch.norm(outputs.F.squeeze(), 2, dim=1)

            squared_l2_norm = l2_norm**2

            sum_norm = torch.sum(squared_l2_norm)

            sum_norm.backward()

            grad_dict_total = {
                k: (grad_dict_total[k] + torch.abs(v.grad))
                for k, v in net.named_parameters()
            }
            N_samples += 1

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            _, pred = torch.max(out.F.squeeze(), 1)
            truepositive = pred * labels.to(device)
            intersection = torch.sum(truepositive == 1)

            mintersection += intersection
            union += torch.sum(pred == 1) + torch.sum(labels == 1) - intersection
            lr = optimizer.param_groups[0]["lr"]

            if tot_iter % 1 == 0 or tot_iter == 1:
                print(
                    f"Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}, mIoU: {(100 * mintersection / union).item()}, lr: {lr} "
                )

                if i % 200 == 0:
                    writer.add_scalar(
                        "Training Loss", accum_loss / accum_iter, tot_iter
                    )
                    writer.add_scalar(
                        "Training mIoU", 100 * mintersection / union, tot_iter
                    )
                    accum_loss, accum_iter, mintersection, union = 0, 0, 0, 0

        if (epoch % 5 == 0) or (epoch == 0):
            torch.save(
                net.state_dict(),
                write_path
                + "weights/exp_14_limited_classes/"
                + config.save_weights
                + "_"
                + str(epoch + 6)
                + ".pth",
            )

    torch.save(
        net.state_dict(),
        write_path
        + "weights/exp_14_limited_classes/"
        + config.save_weights
        + "_"
        + str(epoch + 6)
        + ".pth",
    )

    print("Training Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--weights",
        type=str,
        default="/mnt/d/downloads/KNN/weights/weights/exp_14_limited_classes/weights_exp14_141.pth",
    )

    parser.add_argument("--restrict_training_classes", type=bool, default=True)

    parser.add_argument("--save_weights", type=str, default="weights_exp14")
    parser.add_argument("--save_writer", type=str, default="runs/experiment14")

    parser.add_argument("--backbone", default="mink", type=str)
    parser.add_argument("--dataset", default="kitti", type=str)

    config = parser.parse_args()
    main(config)
