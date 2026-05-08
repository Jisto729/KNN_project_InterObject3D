import os
import concurrent.futures
import numpy as np
from pathlib import Path
from utils.path_resolver import resolve_path_to_root
from utils.directory import parse_directories, parse_files


class KittyParser:
    def __init__(
        self,
        original_dataset,
        crop_margin=5.0,
        target_class=10,
        instance_level=True,
        processed_path="processed_datasets",
        compress=True,
    ):
        self.original_dataset = Path(original_dataset)
        self.crop_margin = crop_margin
        self.target_class = target_class
        self.instance_level = instance_level
        self.compress = compress

        self.processed_path = resolve_path_to_root() / processed_path

    def parse_dataset(self):
        sequences = parse_directories(self.original_dataset / "sequences")
        for sequence in sequences:
            print(sequence)
            self.process_sequence(sequence)

    def process_sequence(self, seq_path):
        bin_files = parse_files(seq_path / "velodyne")
        labels = parse_files(seq_path / "labels")

        if len(bin_files) != len(labels):
            raise ValueError(
                f"Length is different for labels and bin files in {seq_path}"
            )

        max_workers = min(os.cpu_count() or 4, 16)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            executor.map(self.crop_fragment, bin_files, labels)

    def crop_fragment(self, bin_path, label_path):

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        raw_labels = np.fromfile(label_path, dtype=np.uint32)

        semantics = raw_labels & 0xFFFF
        instances = raw_labels >> 16

        valid_mask = (semantics == self.target_class) & (instances > 0)
        valid_instances = np.unique(instances[valid_mask])

        for inst_id in valid_instances:
            obj_points = points[instances == inst_id]
            if len(obj_points) < 100:
                continue

            min_bound = np.min(obj_points, axis=0) - self.crop_margin
            max_bound = np.max(obj_points, axis=0) + self.crop_margin

            crop_mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)

            cropped_points = points[crop_mask]

            if self.instance_level:
                cropped_instances = instances[crop_mask]
                binary_labels = (cropped_instances == inst_id).astype(np.int32)
            else:
                cropped_semantics = semantics[crop_mask]
                binary_labels = (cropped_semantics == self.target_class).astype(
                    np.int32
                )

            save_file(
                self.processed_path,
                bin_path,
                inst_id,
                cropped_points,
                binary_labels,
                self.compress,
            )


def save_file(
    save_path, bin_path, inst_id, cropped_points, binary_labels, compress=True
):
    frame_id = os.path.splitext(os.path.basename(bin_path))[0]
    sequence_id = os.path.basename(os.path.dirname(os.path.dirname(bin_path)))
    full_dir_path = os.path.join(save_path, f"seq{sequence_id}", f"frame{frame_id}")
    os.makedirs(full_dir_path, exist_ok=True)

    if compress:
        npz_filename = os.path.join(full_dir_path, f"inst{inst_id}.npz")

        np.savez_compressed(
            npz_filename,
            points=cropped_points.astype(np.float32),
            labels=binary_labels.astype(np.int32),
        )
    else:
        bin_filename = os.path.join(full_dir_path, f"inst{inst_id}.bin")
        label_filename = os.path.join(full_dir_path, f"inst{inst_id}.label")
        cropped_points.astype(np.float32).tofile(bin_filename)
        binary_labels.astype(np.float32).tofile(label_filename)
