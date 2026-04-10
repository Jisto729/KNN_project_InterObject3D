import numpy as np
import concurrent
import os
from utils.path_resolver import resolve_path_to_root
from utils.directory import parse_directories, parse_files

class KittyClicker():
    def __init__(self,processed_path="processed_datasets"):
        self.processed_path = resolve_path_to_root() + "/" + processed_path

    def parse_dataset(self):
        sequences = parse_directories(self.processed_path)
        for sequence in sequences:
            print(sequence)
            frames = parse_directories(sequence)
            max_workers = min(os.cpu_count() or 4, 16) 
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor.map(self.process_frame, frames)

    def process_frame(self, frame_path):
        archives = parse_files(frame_path)
        for archive in archives:
            with np.load(archive) as data:
                points = data['points']
                labels = data['labels']
            
            T_p, T_n = generate_click_channels(points, labels)
            
            np.savez_compressed(
                archive,
                points=points,
                labels=labels,
                T_p=T_p,
                T_n=T_n
            )
            print(f"Successfully repacked: {archive}")


def generate_click_channels(points, labels, sigma=0.2):
    shape = points.shape[0]
        
    target_indices = np.where(labels == 1)[0]
    if len(target_indices) == 0:
        raise ValueError("No points corresponding to the label")    
    T_p = isolate_clicks(shape, points, target_indices, sigma)
        
    target_points = points[target_indices]
    bounding_box_points_mask = crop_bounding_box_points(target_points, points)
    bg_indices = np.where((labels != 1) & bounding_box_points_mask)[0]
    if len(bg_indices) == 0:
        bg_indices = np.where((labels != 1))[0]
    T_n = isolate_clicks(shape, points, bg_indices, sigma)
            
    return T_p, T_n

def crop_bounding_box_points(object_points, all_points, scaling_factor=1.5):
    min_bound = np.min(object_points, axis=0)
    max_bound = np.max(object_points, axis=0)
    center = (min_bound + max_bound) / 2.0
    dimensions = max_bound - min_bound
    new_min = center - (dimensions * scaling_factor / 2.0)
    new_max = center + (dimensions * scaling_factor / 2.0)
    return np.all((all_points >= new_min) & (all_points <= new_max), axis=1)

def isolate_clicks(shape, points, indices, sigma):
    channel = np.zeros((shape, 1), dtype=np.float32)
    pos_idx = np.random.choice(indices)
    pos_click = points[pos_idx]
            
    dist_sq_pos = np.sum((points - pos_click) ** 2, axis=1)
    channel[:, 0] = np.exp(-dist_sq_pos / (2 * sigma ** 2))
    return channel