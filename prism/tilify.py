import rasterio
import numpy as np
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



'''
MACHINE LEARNING MODEL PIPELINE:
-Tile the Data
-Filer and label tiles, to avoid training on no data
-Dont remove all no fire tiles or 0 signal tiles as model will only learn how to identify "fire zones"
'''

class tilify:
    
    def __init__(self, tile_size=256, stride=256):

        self.tile_size = tile_size
        self.stride = stride  
        self.train_split = 0.7
        self.val_split = 0.2
        self.min_nonzero_pixels = 50  # minimum active pixels in target
        self.tile_coords = {}  # store mapping of tile index to (x, y)
        self.original_shape = None
        self.transform = None

    def tile_raster(self, raster_path):
        with rasterio.open(raster_path) as src:
            tiles = []
            self.tile_coords = {}
            index = 0
            self.original_shape = (src.height, src.width)
            self.transform = src.transform
            for y in range(0, src.height - self.tile_size + 1, self.stride):
                for x in range(0, src.width - self.tile_size + 1, self.stride):
                    window = Window(x, y, self.tile_size, self.tile_size)
                    transform = src.window_transform(window)
                    tile = src.read(window=window)
                    tiles.append((tile, transform))
                    self.tile_coords[index] = (y, x)
                    index += 1
        return tiles

    def gen(self):
        feature_raster_path = f"data/features/uttarakhand_full_stack.tif"
        target_raster_path = f"data/targets/fire_target.tif"
        feature_tiles = self.tile_raster(feature_raster_path)
        target_tiles = self.tile_raster(target_raster_path)

        assert len(feature_tiles) == len(target_tiles)

        X = []
        Y = []
        kept_coords = []

        for idx, ((feat_tile, _), (target_tile, _)) in enumerate(zip(feature_tiles, target_tiles)):
            fire_pixels = np.sum(target_tile > 0)
            if fire_pixels > self.min_nonzero_pixels:
                X.append(feat_tile)
                Y.append(target_tile)
                kept_coords.append(self.tile_coords[idx])
            elif np.random.rand() < 0.25:
                X.append(feat_tile)
                Y.append(target_tile)
                kept_coords.append(self.tile_coords[idx])

        X = np.array(X)
        Y = np.array(Y)

        X_train, X_temp, Y_train, Y_temp, coords_train, coords_temp = train_test_split(
            X, Y, kept_coords, train_size=self.train_split, random_state=42
        )
        X_val, X_test, Y_val, Y_test, coords_val, coords_test = train_test_split(
            X_temp, Y_temp, coords_temp, test_size=1 - self.val_split / (1 - self.train_split), random_state=42
        )

        self.coords = {
            "train": coords_train,
            "val": coords_val,
            "test": coords_test
        }

        return X_train, X_test, X_val, Y_train, Y_test, Y_val, self.coords, self.original_shape, self.tile_size

    def untilify(self, predictions, coords, original_shape, tile_size):
        output = np.zeros(original_shape, dtype=np.float32)
        counts = np.zeros(original_shape, dtype=np.uint8)

        for pred, (y, x) in zip(predictions, coords):
            pred = pred.squeeze()
            output[y:y+tile_size, x:x+tile_size] += pred
            counts[y:y+tile_size, x:x+tile_size] += 1

        counts[counts == 0] = 1
        return output / counts


