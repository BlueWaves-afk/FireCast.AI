import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from datetime import datetime

def clip_and_rasterize_normalized_frp(
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
    print("🔥 Number of fire points on date:", len(fire_gdf))

    # === Reproject fire to boundary CRS if needed
    if fire_gdf.crs != boundary_gdf.crs:
        fire_gdf = fire_gdf.to_crs(boundary_gdf.crs)

    # === Clip fire to boundary
    fire_gdf = gpd.clip(fire_gdf, boundary_gdf)
    print("✅ Fire points after clipping:", len(fire_gdf))

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
        nodata_mask = ref.read(1) == ref.nodata if ref.nodata is not None else np.zeros((ref.height, ref.width), dtype=bool)
        res_x, res_y = ref.res
        print(f"🗺️ CRS: {crs}")
        print(f"📏 Pixel size (resolution): {res_x} x {res_y}")

    # === Reproject buffered fire to raster CRS
    if buffered_fire_gdf.crs != crs:
        buffered_fire_gdf = buffered_fire_gdf.to_crs(crs)

    # === Log-normalize FRP
    if 'FRP' not in buffered_fire_gdf.columns:
        raise ValueError("❌ 'FRP' column not found in fire shapefile.")
    
    # Apply log1p (log(1 + x)) to handle 0 and smooth large values
    buffered_fire_gdf['log_frp'] = np.log1p(buffered_fire_gdf['FRP'])

    # Normalize to [0, 1]
    max_log_frp = buffered_fire_gdf['log_frp'].max()
    if max_log_frp == 0:
        print("⚠️ All FRP values are 0 after log normalization. Skipping.")
        buffered_fire_gdf['norm_frp'] = 0.0
    else:
        buffered_fire_gdf['norm_frp'] = buffered_fire_gdf['log_frp'] / max_log_frp

    # === Create shapes for rasterization
    shapes = list(zip(buffered_fire_gdf.geometry, buffered_fire_gdf['norm_frp']))

    # === Rasterize
    fire_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=output_dtype
    )

    # === Apply nodata mask
    fire_raster = np.where(nodata_mask, 0, fire_raster)

    # === Save the raster
    meta.update({
        'count': 1,
        'dtype': output_dtype,
        'nodata': 0
    })

    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(fire_raster, 1)

    print(f"✅ Log-normalized FRP raster saved to: {output_raster_path}")
    return fire_raster

def find_dates_with_fire(fire_shapefile_path, boundary_shapefile_path):
    fire_dates = []
    no_fire_dates = []

    # Load fire and boundary shapefiles
    fire_gdf = gpd.read_file(fire_shapefile_path)
    boundary_gdf = gpd.read_file(boundary_shapefile_path)

    if 'ACQ_DATE' not in fire_gdf.columns:
        print("❌ ACQ_DATE column not found in fire shapefile.")
        return

    # Convert ACQ_DATE to datetime
    fire_gdf['ACQ_DATE'] = pd.to_datetime(fire_gdf['ACQ_DATE'])

    # Reproject to match CRS if needed
    if fire_gdf.crs != boundary_gdf.crs:
        fire_gdf = fire_gdf.to_crs(boundary_gdf.crs)

    # Clip fire points to boundary
    fire_gdf = gpd.clip(fire_gdf, boundary_gdf)

    # Generate all dates from Jan 1, 2024 to today
    all_dates = pd.date_range(start="2024-01-01", end=datetime.today())

    # Group fire points by date to avoid filtering inside the loop
    fire_dates_set = set(fire_gdf['ACQ_DATE'].dt.date)

    for current_date in all_dates:
        date_only = current_date.date()
        if date_only in fire_dates_set:
            fire_dates.append(date_only.strftime("%Y-%m-%d"))
        else:
            no_fire_dates.append(date_only.strftime("%Y-%m-%d"))

    return fire_dates, no_fire_dates

def display(input_path):

    # Path to daily raster
    raster_path = input_path

    with rasterio.open(raster_path) as src:

        fire_raster = src.read(1)
        plt.imshow(np.clip(fire_raster, 0, 10), cmap="hot")
        plt.title("Fire Raster (FRP)")
        plt.colorbar()
        plt.show()



'''
year = '2024'
month = '01'
day = '02'
date = f'{year}-{month}-{day}' #YYYY-MM-DD

clip_and_rasterize_normalized_frp(
    fire_shapefile_path="data/fire_data/fire_nrt_SV-C2_630938.shp",
    boundary_shapefile_path="data/boundary_files/uttarakhand_boundary_32644.shp",
    reference_raster_path="uttarakhand_static_stack.tif",
    output_raster_path=f"data/targets/fire_target_{date}.tif",
    target_date=date
)


display(f'data/targets/fire_target_{date}.tif')
'''
