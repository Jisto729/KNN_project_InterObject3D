# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import numpy as np
import glob

try:
    import h5py
except:
    print("Install h5py with `pip install h5py`")
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


def stack_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = (
        torch.stack([d["coordinates"] for d in list_data]),
        torch.stack([d["features"] for d in list_data]),
        torch.cat([d["label"] for d in list_data]),
    )

    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


class PointNet(nn.Module):
    def __init__(self, in_channel, out_channel, embedding_channel=1024):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embedding_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embedding_channel)
        self.linear1 = nn.Linear(embedding_channel, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, out_channel, bias=True)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


# MinkowskiNet implementation of a pointnet.
#
# This network allows the number of points per batch to be arbitrary. For
# instance batch index 0 could have 500 points, batch index 1 could have 1000
# points.
class MinkowskiPointNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(3, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiLinear(64, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv5 = nn.Sequential(
            ME.MinkowskiLinear(128, embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel),
            ME.MinkowskiReLU(),
        )
        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.dp1 = ME.MinkowskiDropout()
        self.linear2 = ME.MinkowskiLinear(512, out_channel, bias=True)

    def forward(self, x: ME.TensorField):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)
        x = self.linear1(x)
        x = self.dp1(x)
        return self.linear2(x).F

# MinkowskiNet implementation of a pointnet for segmentation
class MinkowskiPointNetSeg(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        
        # The same as classification
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(in_channel, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiLinear(64, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv5 = nn.Sequential(
            ME.MinkowskiLinear(128, embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel),
            ME.MinkowskiReLU(),
        )
        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        # Segmentation Head
        self.seg_conv1 = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel + 64, 512, bias=False),
            ME.MinkowskiBatchNorm(512), 
            ME.MinkowskiReLU(),
        )
        self.seg_conv2 = nn.Sequential(
            ME.MinkowskiLinear(512, 256, bias=False),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(),
        )
        self.output_linear = ME.MinkowskiLinear(256, out_channel, bias=True)

    def forward(self, x: ME.SparseTensor):
        # Local features that will be concatonated to the global feature
        local_feat = self.conv1(x)  

        # Deeper features
        x = self.conv2(local_feat)  
        x = self.conv3(x)
        x = self.conv4(x)  
        x = self.conv5(x)    
        global_feat = self.max_pool(x)            

        # Concatonating the global feature and the local features
        batch_indices = local_feat.C[:, 0].long()
        global_expanded_feat = global_feat.F[batch_indices]
        combined_feat = torch.cat([local_feat.F, global_expanded_feat], dim=1)
          
        combined = ME.SparseTensor(
            features=combined_feat,
            coordinate_map_key=local_feat.coordinate_map_key,
            coordinate_manager=local_feat.coordinate_manager
        )

        # Segmantation
        x = self.seg_conv1(combined)
        x = self.seg_conv2(x)

        return self.output_linear(x)
    
# MinkowskiNet implementation of a hierarchic pointnet for segmentation
class HierarchicPointNetSeg(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, dimension=3):
        super(HierarchicPointNetSeg, self).__init__(dimension)

        # Encoder
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, 64, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(64, 64),
            ME.MinkowskiReLU()
        )
        self.pool1 = ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=dimension)

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(128, 128, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(128, 128),
            ME.MinkowskiReLU()
        )
        self.pool2 = ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=2, dimension=dimension)

        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(256, 256, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(256, 256),
            ME.MinkowskiReLU()
        )
        self.pool3 = ME.MinkowskiConvolution(256, 512, kernel_size=3, stride=2, dimension=dimension)

        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(512, 512, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(512, 512),
            ME.MinkowskiReLU()
        )
        self.pool4 = ME.MinkowskiConvolution(512, 1024, kernel_size=3, stride=2, dimension=dimension)

        # Global features
        self.global_pool = ME.MinkowskiGlobalMaxPooling()
        # Making global features smaller so that they don't sway the network that much
        self.global_squeeze = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Decoder
        self.up4 = ME.MinkowskiConvolutionTranspose(1024, 1024, kernel_size=3, stride=2, dimension=dimension)
        self.cat4 = nn.Sequential(
            ME.MinkowskiLinear(1024 + 512, 512),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU()
        )

        self.up3 = ME.MinkowskiConvolutionTranspose(512, 512, kernel_size=3, stride=2, dimension=dimension)
        self.cat3 = nn.Sequential(
            ME.MinkowskiLinear(512 + 256, 256),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU()
        )

        self.up2 = ME.MinkowskiConvolutionTranspose(256, 256, kernel_size=3, stride=2, dimension=dimension)
        self.cat2 = nn.Sequential(
            ME.MinkowskiLinear(256 + 128, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU()
        )

        self.up1 = ME.MinkowskiConvolutionTranspose(128, 128, kernel_size=3, stride=2, dimension=dimension)
        self.cat1 = nn.Sequential(
            ME.MinkowskiLinear(128 + 64, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU()
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            ME.MinkowskiLinear(320, 256, bias=False),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(256, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(128, out_channel)
        )

    def forward(self, x: ME.SparseTensor):
        # Encoder
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)

        # Global Features
        g_raw = self.global_pool(p4)
        g_squeezed = self.global_squeeze(g_raw.F)

        # Decoder
        u = self.up4(p4)
        u = ME.cat(u, x4)
        u = self.cat4(u)

        u = self.up3(u)
        u = ME.cat(u, x3)
        u = self.cat3(u)

        u = self.up2(u)
        u = ME.cat(u, x2)
        u = self.cat2(u)

        u = self.up1(u)
        u = ME.cat(u, x1)
        u = self.cat1(u)

        # Global Injection
        batch_indices = u.C[:, 0].long()
        g_expanded = g_squeezed[batch_indices]
        combined_feat = torch.cat([u.F, g_expanded], dim=1)

        final = ME.SparseTensor(
            features=combined_feat,
            coordinate_map_key=u.coordinate_map_key,
            coordinate_manager=u.coordinate_manager
        )

        return self.seg_head(final)
    
# MinkowskiNet implementation of a hierarchic pointnet for segmentation, smaller version
class SmallHierarchicPointNetSeg(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, dimension=3):
        super(SmallHierarchicPointNetSeg, self).__init__(dimension)

        # Encoder
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, 32, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(32, 32),
            ME.MinkowskiReLU()
        )
        self.pool1 = ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=dimension)

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(64, 64, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(64, 64),
            ME.MinkowskiReLU()
        )
        self.pool2 = ME.MinkowskiConvolution(64, 128, kernel_size=3, stride=2, dimension=dimension)

        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(128, 128, kernel_size=3, dimension=dimension),
            ME.MinkowskiLinear(128, 128),
            ME.MinkowskiReLU()
        )
        self.pool3 = ME.MinkowskiConvolution(128, 256, kernel_size=3, stride=2, dimension=dimension)

        # Global features
        self.global_pool = ME.MinkowskiGlobalMaxPooling()
        # Making global features smaller so that they don't sway the network that much
        self.global_squeeze = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Decoder
        self.up3 = ME.MinkowskiConvolutionTranspose(256, 256, kernel_size=3, stride=2, dimension=dimension)
        self.cat3 = nn.Sequential(
            ME.MinkowskiLinear(256 + 128, 128),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU()
        )

        self.up2 = ME.MinkowskiConvolutionTranspose(128, 128, kernel_size=3, stride=2, dimension=dimension)
        self.cat2 = nn.Sequential(
            ME.MinkowskiLinear(128 + 64, 64),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU()
        )

        self.up1 = ME.MinkowskiConvolutionTranspose(64, 64, kernel_size=3, stride=2, dimension=dimension)
        self.cat1 = nn.Sequential(
            ME.MinkowskiLinear(64 + 32, 32),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU()
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            ME.MinkowskiLinear(32+64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiLinear(64, out_channel)
        )

    def forward(self, x: ME.SparseTensor):
        # Encoder
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        # Global Features
        g_raw = self.global_pool(p3)
        g_squeezed = self.global_squeeze(g_raw.F)

        # Decoder
        u = self.up3(p3)
        u = ME.cat(u, x3)
        u = self.cat3(u)

        u = self.up2(u)
        u = ME.cat(u, x2)
        u = self.cat2(u)

        u = self.up1(u)
        u = ME.cat(u, x1)
        u = self.cat1(u)

        # Global Injection
        batch_indices = u.C[:, 0].long()
        g_expanded = g_squeezed[batch_indices]
        combined_feat = torch.cat([u.F, g_expanded], dim=1)

        final = ME.SparseTensor(
            features=combined_feat,
            coordinate_map_key=u.coordinate_map_key,
            coordinate_manager=u.coordinate_manager
        )

        return self.seg_head(final)

class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords *= np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords += np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords

    def __repr__(self):
        return f"Transformation(scale={self.scale_range}, translation={self.trans}, jitter={self.jitter})"


def download_modelnet40_dataset():
    if not os.path.exists("modelnet40_ply_hdf5_2048.zip"):
        print("Downloading the 2k downsampled ModelNet40 dataset...")
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
            ]
        )
        subprocess.run(["unzip", "modelnet40_ply_hdf5_2048.zip"])


class ModelNet40H5(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40h5",
        transform=None,
        num_points=2048,
    ):
        Dataset.__init__(self)
        download_modelnet40_dataset()
        phase = "test" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(data_root, phase)
        self.transform = transform
        self.phase = phase
        self.num_points = num_points

    def load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, "ply_data_%s*.h5" % phase))
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = torch.from_numpy(xyz)
        label = torch.from_numpy(label)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"ModelNet40H5(phase={self.phase}, length={len(self)}, transform={self.transform})"


if __name__ == "__main__":
    dataset = ModelNet40H5(phase="train", data_root="modelnet40_ply_hdf5_2048")
    # Use stack_collate_fn for pointnet
    pointnet_data_loader = DataLoader(
        dataset, num_workers=4, collate_fn=stack_collate_fn, batch_size=16,
    )

    # Use minkowski_collate_fn for pointnet
    minknet_data_loader = DataLoader(
        dataset, num_workers=4, collate_fn=minkowski_collate_fn, batch_size=16,
    )

    # Network
    pointnet = PointNet(in_channel=3, out_channel=20, embedding_channel=1024)
    minkpointnet = MinkowskiPointNet(
        in_channel=3, out_channel=20, embedding_channel=1024, dimension=3
    )

    for i, (pointnet_batch, minknet_batch) in enumerate(
        zip(pointnet_data_loader, minknet_data_loader)
    ):
        # PointNet.
        # WARNING: PointNet inputs must have the same number of points.
        pointnet_input = pointnet_batch["coordinates"].permute(0, 2, 1)
        pred = pointnet(pointnet_input)

        # MinkNet
        # Unlike pointnet, number of points for each point cloud do not need to be the same.
        minknet_input = ME.TensorField(
            coordinates=minknet_batch["coordinates"], features=minknet_batch["features"]
        )
        minkpointnet(minknet_input)
        print(f"Processed batch {i}")