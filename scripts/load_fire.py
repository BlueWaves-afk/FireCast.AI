import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def clip_and_rastaerize_normalized_frp(
    fire_shapefile_path,
    boundary_shapefile_path,
    reference_raster_path,
    output_raster_path,
    target_date,
    buffer_meters=2000,
    output_dtype='float32'
    ):

    # === Load fire shapefile and boundary
    fire_gdf = gpd.read_file(fire_shapefile_path)
    boundary_gdf = gpd.read_file(boundary_shapefile_path)

    # === Filter by date
    if 'ACQ_DATE' in fire_gdf.columns:
        fire_gdf = fire_gdf[fire_gdf['ACQ_DATE'] == target_date]
    logger.info(f"🔥 Number of fire points on date: {len(fire_gdf)}")

    # === Reproject fire to boundary CRS if needed
    if fire_gdf.crs != boundary_gdf.crs:
        fire_gdf = fire_gdf.to_crs(boundary_gdf.crs)

    # === Clip fire to boundary
    fire_gdf = gpd.clip(fire_gdf, boundary_gdf)
    logger.info(f"✅ Fire points after clipping: {len(fire_gdf)}")

    # === Buffer fire points
    buffered_fire_gdf = fire_gdf.copy()
    buffered_fire_gdf["geometry"] = buffered_fire_gdf.geometry.buffer(buffer_meters)

    # === Load reference raster
    with rasterio.open(reference_raster_path) as ref:
        meta = ref.meta.copy()
        transform = ref.transform
        height = ref.height
        width = ref.width
        crs = ref.crs
        nodata_mask = ref.read(1) == ref.nodata if ref.nodata is not None else np.zeros((height, width), dtype=bool)
        res_x, res_y = ref.res
        logger.info(f"🗺️ CRS: {crs}")
        logger.info(f"📏 Pixel size (resolution): {res_x} x {res_y}")

    # === Reproject buffered fire and boundary to raster CRS
    if buffered_fire_gdf.crs != crs:
        buffered_fire_gdf = buffered_fire_gdf.to_crs(crs)
    if boundary_gdf.crs != crs:
        boundary_gdf = boundary_gdf.to_crs(crs)

    # === Log-normalize FRP
    if 'FRP' not in buffered_fire_gdf.columns:
        raise ValueError("❌ 'FRP' column not found in fire shapefile.")
    
    buffered_fire_gdf['log_frp'] = np.log1p(buffered_fire_gdf['FRP'])

    max_log_frp = buffered_fire_gdf['log_frp'].max()
    if max_log_frp == 0:
        logger.info("⚠️ All FRP values are 0 after log normalization. Skipping.")
        buffered_fire_gdf['norm_frp'] = 0.0
    else:
        buffered_fire_gdf['norm_frp'] = buffered_fire_gdf['log_frp'] / max_log_frp

    shapes = list(zip(buffered_fire_gdf.geometry, buffered_fire_gdf['norm_frp']))

    fire_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=output_dtype
    )

    # === Apply nodata mask (input raster nodata)
    fire_raster = np.where(nodata_mask, 0, fire_raster)

    # === Classify into bins
    bins = [0.1, 0.4, 0.6, 0.8]
    fire_class_raster = np.digitize(fire_raster, bins).astype('uint8')  # 0 to 4

    # === Rasterize boundary to mask area of interest
    boundary_mask = rasterize(
        [(geom, 1) for geom in boundary_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # === Assign 255 to areas outside boundary
    fire_class_raster[boundary_mask == 0] = 255

    # === Save the raster
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255
    })

    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(fire_class_raster, 1)

    logger.info(f"✅ Fire class raster saved to: {output_raster_path}")
    
    return fire_class_raster


def save_img(input_path):

    # Path to daily raster
    raster_path = input_path
    
    with rasterio.open(raster_path) as src:

        fire_raster = src.read(1)

        heatmap = (fire_raster * 255).astype('uint8')


        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label="Fire Probability")
        plt.title("Predicted Fire Risk Map")
        plt.axis('off')
        plt.savefig("fire_heatmap_detailed.png", bbox_inches='tight', dpi=300)
        plt.close()

def save_raster(input_path):
    # Load the raster
    with rasterio.open(input_path) as src:
        fire_raster = src.read(1)
        meta = src.meta.copy()  # Copy metadata for reuse

    # Clean invalid values
    fire_raster_cleaned = np.nan_to_num(fire_raster, nan=0.0, posinf=0.0, neginf=0.0)
    fire_raster_cleaned = np.clip(fire_raster_cleaned, 0.0, 1.0)

    # Save cleaned raster as GeoTIFF (float32 format to retain probability)
    meta.update(dtype='float32')

    with rasterio.open(input_path, 'w', **meta) as dst:
        dst.write(fire_raster_cleaned.astype('float32'), 1)

    logger.info(f"✅ Cleaned raster saved to: {os.path.abspath(input_path)}")


'''
save_raster("data/predicted_fire_map.tif")
save_img("data/predicted_fire_map.tif")
'''

'''
display(f'data/targets/fire_target_{date}.tif')
'''
