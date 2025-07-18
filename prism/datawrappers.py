import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob

class FireCastDataset(Dataset):
    def __init__(self, 
                 root_dir=None, 
                 x_array=None, 
                 y_array=None, 
                 transform=None, 
                 augment=True, 
                 to_tensor=True,
                 debug=False):
        """
        Dataset wrapper for fire segmentation.

        Either provide:
        - root_dir: Path containing 'X/' and 'Y/' subfolders with .npy tiles
        OR
        - x_array and y_array directly (in-memory usage)

        Arguments:
        - transform: Optional callable to apply transformation
        - augment: Enable simple flip-based augmentation
        - to_tensor: Convert numpy to torch.Tensor
        - debug: If True, prints target label anomalies
        """
        self.transform = transform
        self.augment = augment
        self.to_tensor = to_tensor
        self.debug = debug

        if x_array is not None and y_array is not None:
            assert len(x_array) == len(y_array), "❌ X and Y must have same length"
            self.x_array = x_array
            self.y_array = y_array
            self.use_memory = True
        elif root_dir:
            self.use_memory = False
            self.x_paths = sorted(glob(os.path.join(root_dir, "X", "*.npy")))
            self.y_paths = sorted(glob(os.path.join(root_dir, "Y", "*.npy")))
            assert len(self.x_paths) == len(self.y_paths), "❌ Number of X and Y tiles must match"
        else:
            raise ValueError("Either root_dir or x_array/y_array must be provided")

    def __len__(self):
        return len(self.x_array) if self.use_memory else len(self.x_paths)

    def __getitem__(self, idx):
        if self.use_memory:
            x = self.x_array[idx]
            y = self.y_array[idx]
        else:
            x = np.load(self.x_paths[idx])
            y = np.load(self.y_paths[idx])

        if self.augment:
            x, y = self.augment_data(x, y)

        if self.transform:
            x, y = self.transform(x, y)

        if self.to_tensor:
            # Clean input
            x = np.nan_to_num(x).copy()
            x = torch.tensor(x, dtype=torch.float32)

            # Clean target
            y = np.nan_to_num(y).astype(np.uint8)
            y = np.where((y >= 0) & (y <= 4), y, 255)  # Keep only classes 0–4 or set to 255
            y = torch.tensor(y, dtype=torch.long)

            if self.debug:
                unique_vals = y.unique()
                unexpected = unique_vals[~torch.isin(unique_vals, torch.tensor([0, 1, 2, 3, 4, 255]))]
                if len(unexpected) > 0:
                    print(f"🚨 Unexpected label values at index {idx}: {unique_vals}")

        return x, y

    def augment_data(self, x, y):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=1).copy()  # Flip horizontally
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=0).copy()  # Flip vertically
        return x, y
