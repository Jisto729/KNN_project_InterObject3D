import numpy as np
from torch.utils.data import Dataset
from utils.directory import parse_directories
from utils.path_resolver import resolve_path_to_root
from kitty_clicker import generate_click_channels, KittyClicker
from kitty_parser import KittyParser


def preprocess_data(path):
    processed_dir = resolve_path_to_root() / "processed_datasets"
    
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    elif any(processed_dir.iterdir()):
        print(f"Directory {processed_dir} is already populated. Skipping preprocessing.")
        return
    
    parser = KittyParser(path)
    parser.parse_dataset()
    clicker = KittyClicker()
    clicker.parse_dataset()

class KittiDataset(Dataset):
    
    def __init__(self):  
        self.archives = []
        processed_datasets = parse_directories(resolve_path_to_root()/"processed_datasets")
        for sequence in processed_datasets:
            for frame in parse_directories(sequence):
                self.archives.extend(parse_directories(frame))

    def __len__(self):
        return len(self.archives)

    def __getitem__(self, key):
        try:
            with np.load(self.archives[key]) as data:
                points = data['points']
                labels = data['labels']

                if 'T_p' in data and 'T_n' in data:
                    T_p = data['T_p']
                    T_n = data['T_n']
                else:
                    T_p, T_n = generate_click_channels(points, labels)
        except Exception as e:
            print(f"\n[WARNING] Skipping corrupted file: {self.archives[key]}")
            print(f"Error details: {e}")
            random_fallback_index = np.random.randint(0, len(self.archives))
            return self.__getitem__(random_fallback_index)

        feats = np.column_stack((points, T_p, T_n))
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=points/0.05,
            features=feats,
            labels=labels,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels

