
from torch.utils.data import Dataset
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import torch
import os
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm


'''
MACHINE LEARNING MODEL PIPELINE:
-Tile the Data
-Filer and label tiles, to avoid training on no data
-Dont remove all no fire tiles or 0 signal tiles as model will only learn how to identify "fire zones"
'''

class tilify():

    def __init__(self,date1,date2):
        self.feature_raster_path = f"data/features/uttarakhand_full_stack_{date1}.tif"
        self.target_raster_path = f"data/targets/fire_target_{date2}.tif"
        self.tile_size = 256
        self.stride = 256  # You can use smaller stride for overlapping tiles
        self.output_base_dir = "data/tiles"
        self.train_split = 0.7
        self.val_split = 0.2  # test split is 1 - (train + val)
        self.min_nonzero_pixels = 50  # retain tiles with this many positive pixels at minimum

    # Utility to tile a raster and return tile data and metadata
    def tile_raster(self, raster_path, tile_size, stride):
        with rasterio.open(raster_path) as src:
            tiles = []
            for y in range(0, src.height - tile_size + 1, stride):
                for x in range(0, src.width - tile_size + 1, stride):
                    window = Window(x, y, tile_size, tile_size)
                    transform = src.window_transform(window)
                    tile = src.read(window=window)
                    tiles.append((tile, transform))
        return tiles

    # Create directory structure
    def prepare_dirs(self,base_dir):
        for subset in ['train', 'val', 'test']:
            for kind in ['features', 'targets']:
                path = os.path.join(base_dir, subset, kind)
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

    # Save tile as .npy
    def save_tile(self, tile_data, out_path):
        np.save(out_path, tile_data)

    def gen(self):

        # Tiling both rasters
        feature_tiles = self.tile_raster(self.feature_raster_path, self.tile_size, self.stride)
        target_tiles = self.tile_raster(self.target_raster_path, self.tile_size, self.stride)

        # Sanity check: number of tiles should match
        assert len(feature_tiles) == len(target_tiles)

        # Balance tiles
        X = []
        Y = []
        for (feat_tile, _), (target_tile, _) in zip(feature_tiles, target_tiles):
            fire_pixels = np.sum(target_tile > 0)
            if fire_pixels > self.min_nonzero_pixels:
                X.append(feat_tile)
                Y.append(target_tile)
            elif np.random.rand() < 0.25:  # include some no-fire tiles
                X.append(feat_tile)
                Y.append(target_tile)

        X = np.array(X)
        Y = np.array(Y)

        # Split dataset
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=self.train_split, random_state=42)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=1 - self.val_split/(1 - self.train_split), random_state=42)

        # Save tiles
        '''
        self.prepare_dirs(self.output_base_dir)

        def save_split(split_name, features, targets):
            for i, (f, t) in enumerate(zip(features, targets)):
                f_path = os.path.join(self.output_base_dir, split_name, "features", f"{i:04d}.npy")
                t_path = os.path.join(self.output_base_dir, split_name, "targets", f"{i:04d}.npy")
                self.save_tile(f, f_path)
                self.save_tile(t, t_path)

        save_split("train", X_train, Y_train)
        save_split("val", X_val, Y_val)
        save_split("test", X_test, Y_test)
        '''


        # Summary
        tile_counts = {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        }
        print(tile_counts)

        return X_train, X_test, X_val, Y_train, Y_test, Y_val 



