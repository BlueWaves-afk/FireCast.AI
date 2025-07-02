import cdsapi
import os
import xarray as xr
import numpy as np
import zipfile
import glob
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def display_multiband_raster(raster_path, band_indices=None, titles=None, cmap='viridis'):
    with rasterio.open(raster_path) as src:
        count = src.count
        print(f"📊 Number of bands: {count}")

        if band_indices is None:
            band_indices = list(range(1, count + 1))  # 1-based indexing in rasterio

        fig, axes = plt.subplots(1, len(band_indices), figsize=(6 * len(band_indices), 6))

        if len(band_indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, band_indices):
            band = src.read(idx)
            band = np.where(band == src.nodata, np.nan, band)

            im = ax.imshow(band, cmap=cmap)
            title = f'Band {idx}' if not titles else titles[idx - 1]
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()
        plt.show()

def convert_nc_to_multiband_tif(nc_path, output_dir):

    ds = xr.open_dataset(nc_path, engine="netcdf4")

    # Use 'valid_time' instead of 'time'
    if 'valid_time' not in ds:
        raise ValueError("❌ No valid_time dimension in dataset.")

    # Select the first valid_time (you can adjust this logic)
    selected_time = ds['valid_time'].values[0]
    print(f"🕛 Selected timestamp: {selected_time}")

    bands = []
    band_names = []

    for var in ds.data_vars:
        da = ds[var]

        # Skip coords and non-spatial vars
        if {'longitude', 'latitude'}.issubset(da.dims):
            # Select data at valid_time
            da = da.sel(valid_time=selected_time)
            da = da.squeeze()

            # Set spatial dims and CRS
            da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            da.rio.write_crs("EPSG:4326", inplace=True)

            bands.append(da.values.astype(np.float32))
            band_names.append(var)

    if not bands:
        raise ValueError("❌ No valid variables with lat/lon dimensions found.")

    # Stack into one 3D array: (bands, height, width)
    stacked = np.stack(bands, axis=0)

    # Use first DA for metadata
    template = ds[band_names[0]].sel(valid_time=selected_time).squeeze()
    transform = template.rio.transform()
    height, width = template.shape

    out_meta = {
        "driver": "GTiff",
        "count": len(bands),
        "dtype": "float32",
        "height": height,
        "width": width,
        "crs": "EPSG:4326",
        "transform": transform
    }

    # Output path
    out_filename = f"weather_{str(selected_time)[:10]}.tif"
    out_path = os.path.join(output_dir, out_filename)

    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(stacked)
    
    os.remove(nc_path)
    print(f"✅ Saved multiband weather raster: {out_path}")
    return out_path

def preprocess_and_normalize_multiband_raster(
    input_tif_path,
    boundary_shapefile_path,
    reference_raster_path,
    output_path,
    nodata_value=np.nan
    ):
    # === Load boundary
    boundary_gdf = gpd.read_file(boundary_shapefile_path)
    with rasterio.open(reference_raster_path) as ref_raster:
        ref_crs = ref_raster.crs
        ref_transform = ref_raster.transform
        ref_height = ref_raster.height
        ref_width = ref_raster.width

    boundary_gdf = boundary_gdf.to_crs(ref_crs)
    geometry = [feature["geometry"] for feature in boundary_gdf.__geo_interface__["features"]]

    # === Read source raster
    with rasterio.open(input_tif_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_meta = src.meta
        data = src.read()
        band_count = src.count

    # === Reproject to match reference raster
    reprojected_data = np.empty((band_count, ref_height, ref_width), dtype=np.float32)

    for i in range(band_count):
        reproject(
            source=data[i],
            destination=reprojected_data[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

    # === Clip to boundary
    temp_meta = src_meta.copy()
    temp_meta.update({
        'crs': ref_crs,
        'transform': ref_transform,
        'width': ref_width,
        'height': ref_height,
        'count': band_count
    })

    with rasterio.open("data/temp/weather/temp.tif", "w", **temp_meta) as temp_dst:
        temp_dst.write(reprojected_data)

    with rasterio.open("data/temp/weather/temp.tif") as clip_src:
        clipped_data, clipped_transform = mask(clip_src, geometry, crop=True, nodata=nodata_value)
        new_meta = clip_src.meta.copy()
        new_meta.update({
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
            "nodata": nodata_value,
            "dtype": "float32"
        })

    # === Normalize each band based on assumed order of ERA5 variables
    # [0] 2m_temperature (K), [1] 2m_dewpoint_temperature (K), [2] 10m_u_wind, [3] 10m_v_wind

    def normalize_band(band, idx):
        if idx in [0, 1]:  # temperature bands: convert to °C and normalize [-20, 50] → [0, 1]
            celsius = band - 273.15
            return np.clip((celsius + 20) / 70, 0, 1)
        elif idx in [2, 3]:  # wind: normalize [-20, 20] → [0, 1]
            return np.clip((band + 20) / 40, 0, 1)
        else:
            return band  # leave unchanged if unknown

    normalized = np.stack([normalize_band(clipped_data[i], i) for i in range(clipped_data.shape[0])])

    # === Save final output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **new_meta) as dst:
        dst.write(normalized.astype(np.float32))

    print(f"✅ Preprocessed, normalized, and aligned raster saved to: {output_path}")

def extract(file_path):
    extract_dir = "data/temp/weather"

    # Step 1: Extract the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Step 2: Find files
    instant_files = glob.glob(os.path.join(extract_dir, "*instant.nc"))
    accum_files = glob.glob(os.path.join(extract_dir, "*accum.nc"))

    # Step 3: Delete accum.nc files
    for f in accum_files:
        os.remove(f)

    # Step 4: Delete the zip file
    os.remove(file_path)

    # Step 5: Return the path to instant.nc
    if instant_files:
        return instant_files[0]
    else:
        raise FileNotFoundError("❌ No '*instant.nc' file found in the archive.")

def weather_data(date):
    dt = datetime.strptime(date, '%Y-%m-%d')
    
    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature',
                '2m_dewpoint_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                'total_precipitation'
            ],
            'year': f'{year}',
            'month': f'{month}',
            'day': [f'{day}'],
            'time': ['12:00'],
            'area': [34.0, 74.5, 27.0, 82.0],
        },
        'era5_sample.nc'
    )

'''
year = "2024"
month = "01"
day="01"

weather_data(year,month,day)

path = extract("era5_sample.nc")
path = convert_nc_to_multiband_tif(path,'data/weather')

preprocess_and_normalize_multiband_raster(
    input_tif_path=path,
    boundary_shapefile_path="data/boundary_files/uttarakhand_boundary_32644.shp",
    reference_raster_path="uttarakhand_full_stack.tif",
    output_path=path
)
prepare_dataset.stack_weather_with_static(
    static_raster_path="uttarakhand_static_stack.tif",
    weather_raster_path= f"data/weather/weather_{year}-{month}-{day}.tif",
    output_path= f"data/features/uttarakhand_full_stack_{year}-{month}-{day}.tif"
)

'''


'''
display_multiband_raster(
    raster_path=path,
    band_indices=[1, 2, 3, 4],
    titles=["Temperature", "Dew Point", "Wind U", "Wind V"]
)

'''

