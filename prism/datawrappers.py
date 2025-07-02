import torch
from torch.utils.data import Dataset
import numpy as np
import os
import rasterio

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

class FireCastDataset(Dataset):
    def __init__(self, 
                 root_dir=None, 
                 x_array=None, 
                 y_array=None, 
                 transform=None, 
                 augment=True, 
                 to_tensor=True):
        """
        root_dir: path to folder (should contain 'X' and 'Y' subfolders) OR
        x_array, y_array: pass NumPy arrays directly
        """
        self.transform = transform
        self.augment = augment
        self.to_tensor = to_tensor

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

        # Optional data augmentation
        if self.augment:
            x, y = self.augment_data(x, y)

        # Apply transform if provided
        if self.transform:
            x, y = self.transform(x, y)

        # Convert to torch tensors
        if self.to_tensor:
            x = np.nan_to_num(x).copy()
            y = np.nan_to_num(y).copy()
            x = torch.tensor(x.copy(), dtype=torch.float32)
            y = torch.tensor(y.copy(), dtype=torch.float32)

        return x, y

    def augment_data(self, x, y):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        return x, y
