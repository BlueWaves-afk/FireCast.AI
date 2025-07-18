from . import tilify
from torch.utils.data import DataLoader
from .datawrappers import FireCastDataset
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.optim as optim
from . import trainingroutine
import torch
import os 
import numpy as np
from tqdm import tqdm  
import rasterio
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FireCastPRISMModel():

    def __init__(self):
        logger.info("Initializing PRISMv1 Model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.get_model()

    def get_loss(self):
        #account for class frequency imbalance
        #class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)  # Adjust
        frequencies = np.array([0.999007, 0.000295, 0.000323, 0.000244, 0.000132])
        weights = 1.0 / frequencies #inverse frequency weighting
        weights = weights / weights.sum() #normalize weights to sum to 1
        #convert to tensor and move to device
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device.type)
        #clipping large weights to prevent overflow
        weights = np.clip(weights, a_min=None, a_max=1000)
        
        return nn.CrossEntropyLoss(weight=class_weights,ignore_index=255) # Ignore index for no-data pixels
        
    def get_optim(self):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return optimizer, scheduler

    def get_dataset(self):
        t = tilify.tilify(tile_size=128, stride=128)
        logger.info("Tiling raster data...")
        X_train, X_test, X_val, Y_train, Y_test, Y_val, _, _, _ = t.gen()

        logger.info("Splitting data into train, validation, and test sets...")
        train_dataset = FireCastDataset(x_array=X_train, y_array=Y_train)
        val_dataset = FireCastDataset(x_array=X_val, y_array=Y_val)
        train_loader = DataLoader(train_dataset, pin_memory=True, num_workers=4, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, pin_memory=True, num_workers=4, batch_size=8, shuffle=True)

        return train_loader, val_loader

    def get_model(self):
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=9,
            classes=5,
            activation=None
        )

        if os.path.exists('best_model.pth'):
            logger.info("Loading pre-trained model weights...")
            model.load_state_dict(torch.load('best_model.pth', map_location=self.device,weights_only=True))
            model.to(self.device)
            logger.info("Model weights loaded successfully.")
        else:
            logger.warning("Model weights not found. Initializing a new model.")

        return model

    def train(self):
        loss_fn = self.get_loss()
        train_loader, val_loader = self.get_dataset()
        optimizer, scheduler = self.get_optim()
        trainingroutine.FireCastTrainer(self.model, train_loader, val_loader, loss_fn, optimizer, scheduler)

    def save_prediction_geotiff(self, predicted_map, reference_raster_path, output_path="predictions/predicted_fire_map.tif"):
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(reference_raster_path) as ref:
            meta = ref.meta.copy()

        meta.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": 255
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(predicted_map.astype("uint8"), 1)

        logger.info(f"✅Saved fire risk class GeoTIFF to: {output_path}")

    def predict(self, raster_path):
        t = tilify.tilify(tile_size=128, stride=128)
        tile_data = t.tile_raster(raster_path)
        logger.info('✅ Done tilifying data...')

        tiles = [tile for tile, _ in tile_data]
        coords = [t.tile_coords[i] for i in range(len(tile_data))]
        H, W = t.original_shape
        tile_size = t.tile_size

        self.model.eval()
        self.model.to(self.device, memory_format=torch.channels_last)

        inputs = torch.tensor(np.array(tiles, dtype=np.float32), dtype=torch.float32)
        assert inputs.ndim == 4, f"Expected 4D input, got {inputs.shape}"
        inputs = inputs.to(self.device, memory_format=torch.channels_last)

        logger.info('✅ Converted tiles to torch tensor...')

        preds = []
        batch_size = 8

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for i in tqdm(range(0, len(inputs), batch_size), desc="🔍 Predicting on tiles"):
                batch = inputs[i:i + batch_size]
                outputs = self.model(batch)  # Shape: (B, 5, H, W)
                predicted_classes = torch.argmax(outputs, dim=1)  # Shape: (B, H, W)
                preds.extend(predicted_classes.cpu().numpy())

        logger.info("✅ Done predicting. Total tiles predicted.")

        predicted_map = t.untilify(preds, coords, (H, W), tile_size)
        #predicted_map = gaussian_filter(predicted_map, sigma=1)
        with rasterio.open(raster_path) as ref:
            nodata_value = ref.nodata
            if nodata_value is not None:
                boundary_mask = ref.read(1) == nodata_value
            else:
                boundary_mask = np.zeros((ref.height, ref.width), dtype=bool)
            
        predicted_map[boundary_mask] = 255
        self.save_prediction_geotiff(predicted_map, reference_raster_path=raster_path)

        del inputs, outputs, preds
        torch.cuda.empty_cache()

        logger.info("✅ Done untilifying and saving prediction GeoTIFF.")
