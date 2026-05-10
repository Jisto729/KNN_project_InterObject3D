import torch
import numpy as np
import glob
import os
import open3d as o3d
import MinkowskiEngine as ME
from torch.utils.data import Dataset
import pickle
import io
from pathlib import Path
from examples.minkunet import MinkUNet34C, MinkUNet18B
from examples.pointnet import (
    MinkowskiPointNetSeg,
    HierarchicPointNetSeg,
    SmallHierarchicPointNetSeg,
)


class RandomLineDatasetSemKITTINPZ(Dataset):
    def __init__(self, config):
        folder_path = config.dataset_scenes
        self.archives = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
        self.quantization_size = 0.05
        self.dataset_size = len(self.archives)
        print(self.dataset_size)

        model_name = getattr(config, "used_model", getattr(config, "model_used", ""))
        self.use_two_channels = (
            model_name == "HierarchicPointNetSeg"
        ) or model_name == "SmallHierarchicPointNetSeg"

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, key):
        try:
            npz_path = self.archives[key]
            with np.load(npz_path) as data:
                coords = data["points"]
                labels = data["labels"]

            scenecolors = np.zeros((coords.shape[0], 3))
            T_p = np.zeros(coords.shape[0])
            T_n = np.zeros(coords.shape[0])

            if self.use_two_channels:
                feats = np.column_stack((T_p, T_n))  # 2 channels
            else:
                feats = np.column_stack((scenecolors, T_p, T_n))  # 5 channels

            _, inverse_map = ME.utils.sparse_quantize(
                coordinates=coords,
                quantization_size=self.quantization_size,
                return_index=True,
                ignore_label=-100,
            )

            coords_qv = coords[inverse_map]
            feats_qv = feats[inverse_map]

            if labels.ndim > 1 and labels.shape[1] > 1:
                labels_qv = labels[:, 0][inverse_map]
            else:
                labels_qv = labels.reshape(-1)[inverse_map]

            gtcolors = labels.reshape(-1, 1)

            file_stem = Path(npz_path).stem
            scene_name = file_stem
            object_id = str(key)
            return (
                scene_name,
                object_id,
                coords,
                scenecolors,
                gtcolors,
                T_p,
                T_n,
                feats,
                coords_qv,
                feats_qv,
                labels_qv,
                inverse_map,
            )

        except Exception as e:
            print(f"\n[WARNING] Skipping corrupted file: {self.archives[key]}")
            print(f"Error details: {e}")
            random_fallback_index = np.random.randint(0, self.dataset_size)
            return self.__getitem__(random_fallback_index)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class InteractiveSegmentationModel(object):
    def __init__(
        self,
        pretraining_weights="weights/weights_exp14_11_pointnet.pth",
    ):

        self.pretraining_weights_file = pretraining_weights

    def create_model(self, device, used_model, pretrained_weights_file=None):

        if used_model == "MinkUNet34C":
            model = MinkUNet34C(in_channels=5, out_channels=2, D=3).to(device)
        elif used_model == "MinkUNet18B":
            model = MinkUNet18B(in_channels=5, out_channels=2, D=3).to(device)
        elif used_model == "MinkowskiPointNetSeg":
            model = MinkowskiPointNetSeg(in_channel=5, out_channel=2, dimension=3).to(
                device
            )
        elif used_model == "HierarchicPointNetSeg":
            model = HierarchicPointNetSeg(in_channel=2, out_channel=2, dimension=3).to(
                device
            )
        elif used_model == "SmallHierarchicPointNetSeg":
            model = SmallHierarchicPointNetSeg(
                in_channel=2, out_channel=2, dimension=3
            ).to(device)

        if pretrained_weights_file:
            #  Get weights
            weights = pretrained_weights_file
            print("weights", weights)
            if pretrained_weights_file:
                #  Get weights
                if not torch.cuda.is_available():
                    map_location = "cpu"
                    print("Cuda not found, using CPU")
                    model_dict = torch.load(weights, map_location)

                else:
                    map_location = None
                    model_dict = torch.load(weights, map_location)
            model.load_state_dict(model_dict)
        return model

    def add_optimiser(self, model, config):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        return criterion, optimizer

    def create_model_input(
        self, scenecolors, pccolors, nccolors, use_training_clicks=False
    ):
        feats = np.column_stack((scenecolors, pccolors[:, 0], nccolors[:, 0]))
        if not use_training_clicks:
            feats[:, 3:5] = np.zeros((feats.shape[0], 2))
        else:
            print("*Using training clicks for debug!*")
        return feats

    def prediction(self, feats, coords, model, device):
        # with torch.no_grad():
        voxel_size = 0.05
        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            features=feats,
            coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=device,
        )
        model.eval()
        logits = model(sinput).slice(sinput)
        # get the prediction on the input tensor field
        logits = logits.F
        _, pred = logits.max(1)

        return pred, logits

    def confidence(self, logits, pred):
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Extract the probability specifically for the Object
        object_probs = probabilities[:, 1]

        # Isolate ONLY the points the model actually predicted as the object
        predicted_object_probs = object_probs[pred == 1]

        # average confidence

        if len(predicted_object_probs) > 0:
            mask_confidence = predicted_object_probs.mean().item()
        else:
            mask_confidence = 0.0

        return mask_confidence

    def mean_iou(self, pred, labels):
        intersection = labels * pred
        truepositive = intersection.sum()
        union = torch.logical_or(labels, pred)
        union = torch.sum(union)
        iou = 100 * (truepositive / union)
        return iou

    def visualize_open3d(self, coords, feats, click, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector((feats[:, 0:3]).cpu().numpy())

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 1.0, 1.0])

        l = torch.min(coords[labels == 1], axis=0).values.cpu().numpy()
        m = torch.max(coords[labels == 1], axis=0).values.cpu().numpy()

        bbox1 = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=l - 0.5, max_bound=m + 0.5
        )

        bbox2 = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=click - 0.05, max_bound=click + 0.05
        )
        bbox2.color = (1, 0, 0)
        bbox3 = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=click - 0.025, max_bound=click + 0.025
        )
        bbox3.color = (0, 1, 0)
        bbox4 = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=click - 0.005, max_bound=click + 0.005
        )
        bbox4.color = (0, 0, 1)

        pcd_crop = pcd.crop(bbox1)

        return pcd_crop, pcd.normals

    def pick_points(self, pcd, prediction):
        print("")
        print("1) Please pick one correspondence using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")

        import open3d.visualization.gui as gui

        vis = o3d.visualization.VisualizerWithEditing()

        vis.create_window()
        vis.get_render_option().point_size = 10.5

        vis.add_geometry(prediction)

        vis.run()
        vis.destroy_window()
        print("")

        if len(vis.get_picked_points()) == 2:
            return vis.get_picked_points()[1], 0
        else:
            return vis.get_picked_points()[0], 1

    def get_next_click_coo_torch_real_user(self, coords, pred, gt, feats, num_clicks):

        ## only 2 channels
        num_channels = feats.size()[1]
        if num_channels >= 3:
            base_colors = feats[:, 0:3].cpu().numpy()
        else:
            # 2 channels = no RGB. Create dummy grayscale colors for Open3D
            base_colors = np.full((feats.shape[0], 3), 0.5)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(base_colors)

        prediction = o3d.geometry.PointCloud()
        pred_rgb = np.zeros(np.shape(coords[:, 0:3].cpu().numpy()))
        if num_clicks != 0:
            pred_rgb[:, 1] = 255 * (pred.cpu().numpy())
        prediction.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        prediction.colors = o3d.utility.Vector3dVector(pred_rgb + base_colors)

        picked_point, gt_point = self.pick_points(pcd, prediction)
        if not picked_point:
            return None, None
        center_id = picked_point

        return coords[center_id], gt_point

    def get_next_click_coo_torch(self, discrete_coords, unique_labels, gt, pred):

        zero_indices = unique_labels == 0  # background
        one_indices = unique_labels == 1  # foreground
        if zero_indices.sum() == 0 or one_indices.sum() == 0:
            return None, None, -1, None, None

        # All distances from foreground points to background points
        pairwise_distances = torch.cdist(
            discrete_coords[zero_indices.cpu(), :],
            discrete_coords[one_indices.cpu(), :],
        )
        # Bg points on the border
        pairwise_distances, _ = torch.min(pairwise_distances, dim=0)
        # point furthest from border
        center_id = torch.where(
            pairwise_distances == torch.max(pairwise_distances, dim=0)[0]
        )
        center_coo = discrete_coords[one_indices.cpu(), :][center_id[0][0]]
        center_label = gt[one_indices.cpu()][center_id[0][0]]

        candidates = discrete_coords[one_indices.cpu(), :]
        candidates_heat = []
        max_dist = torch.max(pairwise_distances)

        for i in pairwise_distances:
            candidates_heat.append([255 * i / max_dist, 0, 0])
        return center_coo, center_label, max_dist, candidates, candidates_heat

    def generate_clickmask_torch(self, vertices_pc, center_coo, cubeedge=0.05):

        refx, refy, refz = center_coo

        vertices_pc[
            torch.logical_and(
                torch.logical_and(
                    (torch.abs(vertices_pc[:, 0] - refx) < cubeedge),
                    (torch.abs(vertices_pc[:, 1] - refy) < cubeedge),
                ),
                (torch.abs(vertices_pc[:, 2] - refz) < cubeedge),
            ),
            3,
        ] = 1

        return vertices_pc[:, 3].unsqueeze_(1)

    def sample_user_input(self, user_input, coords, feats):
        n_points = len(user_input)
        sampled_input = torch.zeros((feats.shape[0], 1))
        if n_points > 1:
            nsamples = np.random.randint(1, n_points)
            sample_indices = np.random.choice(range(n_points), nsamples, replace=False)
            sampled_points = np.array(user_input)[sample_indices]
            for i in range(nsamples):
                sampled_input += self.generate_clickmask_torch(
                    torch.hstack(
                        (
                            coords[:, 0:3],
                            torch.zeros((feats.shape[0], 1)) * 255,
                        )
                    ),
                    torch.tensor(sampled_points[i]),
                )
        elif n_points == 1:
            sampled_input += self.generate_clickmask_torch(
                torch.hstack(
                    (
                        coords[:, 0:3],
                        torch.zeros((feats.shape[0], 1)) * 255,
                    )
                ),
                torch.tensor(user_input[0]),
            )
        else:
            return sampled_input
        return sampled_input

    def get_next_simulated_click_dense(self, pred, labels, coords, inseg_model_class):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP
        # get next positive click candidate
        pcenter_coo, pcenter_gt, pmax_dist, candidates_p, candidates_p_heat = (
            inseg_model_class.get_next_click_coo_torch(coords, fn, labels, pred)
        )
        # get next negative click candidate
        ncenter_coo, ncenter_gt, nmax_dist, candidates_n, candidates_n_heat = (
            inseg_model_class.get_next_click_coo_torch(coords, fp, labels, pred)
        )
        if pmax_dist >= nmax_dist:
            center_coo = pcenter_coo
            center_gt = pcenter_gt
            candidates = candidates_p
            candidates_heat = candidates_p_heat
        else:
            center_coo = ncenter_coo
            center_gt = ncenter_gt
            candidates = candidates_n
            candidates_heat = candidates_n_heat
        return center_coo, center_gt, candidates, candidates_heat, fn, fp

    def get_next_simulated_click(
        self, pred, labels, coords_qv, labels_qv, inverse_map, inseg_model_class
    ):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP

        # get next positive click candidate
        pcenter_coo, pcenter_gt, pmax_dist, candidates_p, candidates_p_heat = (
            inseg_model_class.get_next_click_coo_torch(
                coords_qv, fn[inverse_map], labels_qv, pred[inverse_map]
            )
        )
        # get next negative click candidate
        ncenter_coo, ncenter_gt, nmax_dist, candidates_n, candidates_n_heat = (
            inseg_model_class.get_next_click_coo_torch(
                coords_qv, fp[inverse_map], labels_qv, pred[inverse_map]
            )
        )
        if pmax_dist >= nmax_dist:
            center_coo = pcenter_coo
            center_gt = pcenter_gt
            candidates = candidates_p
            candidates_heat = candidates_p_heat
        else:
            center_coo = ncenter_coo
            center_gt = ncenter_gt
            candidates = candidates_n
            candidates_heat = candidates_n_heat

        return center_coo, center_gt, candidates, candidates_heat, fn, fp

    def get_next_uncertain_click(self, pred, labels, coords, logits, used_pixels):
        fn = torch.logical_and(torch.logical_xor(pred, labels), labels)  # FN
        fp = torch.logical_and(torch.logical_xor(pred, labels), pred)  # FP

        correct = torch.logical_and(labels, pred)
        unca = torch.logits[:, 0] - logits[:, 1]
        unca[correct] = 1000

        for used_pixel in used_pixels:
            unca[used_pixel] = 1000
            refx, refy, refz = coords[used_pixel][0]
            cubeedge = 0.05

            unca[
                torch.logical_and(
                    torch.logical_and(
                        (torch.abs(coords[:, 0] - refx) < cubeedge),
                        (torch.abs(coords[:, 1] - refy) < cubeedge),
                    ),
                    (torch.abs(coords[:, 2] - refz) < cubeedge),
                )
            ] = 1000

        k1 = 1000
        candidate_vals, candidate_ids = torch.topk(unca, k=k1, largest=False)
        best_candidate = candidate_ids[torch.randint(1, k1, (1, 1))][0]
        used_pixels.append(best_candidate)
        center_coo = coords[best_candidate[0]]
        center_gt = labels[best_candidate[0]]

        print(unca[best_candidate])
        print(logits[best_candidate, 0], logits[best_candidate, 1])

        candidates = coords[candidate_ids, :]
        candidates_heat = []
        max_dist = torch.max(candidate_vals)

        for i in candidate_vals:
            candidates_heat.append([255 * i.item() / max_dist.item(), 0, 0])
        return center_coo, center_gt, candidates, candidates_heat, fn, fp


class InteractiveSegmentationModelAdaptive(InteractiveSegmentationModel):
    def __init__(
        self,
        pretraining_weights="weights/weights_exp14_11_pointnet.pth",
    ):

        self.pretraining_weights_file = pretraining_weights
        InteractiveSegmentationModel.__init__(self, pretraining_weights)

    def create_model(self, device, config, pretrained_weights_file=None):
        model = MinkUNet34C(in_channels=5, out_channels=2, D=3).to(device)
        if pretrained_weights_file:
            #  Get weights
            weights = pretrained_weights_file
            if not torch.cuda.is_available():
                map_location = "cpu"

            else:
                map_location = None
            model_dict = torch.load(weights, map_location)
            model.load_state_dict(model_dict)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

        return model, criterion, optimizer

    def prediction(self, feats, coords, model, device):
        voxel_size = 0.05

        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            features=feats,
            coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=device,
        )

        logits = model(sinput).slice(sinput)

        logits = logits.F
        _, pred = logits.max(1)

        return pred, logits
