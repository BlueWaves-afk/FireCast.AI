import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import load_weather
import load_fire
from datetime import datetime, timedelta
import shutil
from prism.model import FireCastModel

def remove_bands_from_raster(input_raster_path, output_raster_path, bands_to_remove):
    """
    Removes specified band indices (1-based) from a multi-band raster.

    Parameters:
        input_raster_path (str): Path to the input stacked raster.
        output_raster_path (str): Path to save the output raster with selected bands removed.
        bands_to_remove (list): List of band indices (1-based) to remove.
    """
    with rasterio.open(input_raster_path) as src:
        all_indices = list(range(1, src.count + 1))
        keep_indices = [i for i in all_indices if i not in bands_to_remove]

        print(f"📦 Original Bands: {all_indices}")
        print(f"❌ Removing Bands: {bands_to_remove}")
        print(f"✅ Keeping Bands: {keep_indices}")

        # Read only the bands to keep
        data = np.stack([src.read(i) for i in keep_indices])

        # Update metadata
        meta = src.meta.copy()
        meta.update({
            "count": len(keep_indices),
            "dtype": data.dtype,
        })

        os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
        with rasterio.open(output_raster_path, "w", **meta) as dst:
            dst.write(data)

    print(f"✅ New raster saved to: {output_raster_path}")

def display_multiband_raster(raster_path, band_indices=None, titles=None, cmap='viridis', normalize=True):
    with rasterio.open(raster_path) as src:
        count = src.count
        print(f"📊 Number of bands: {count}")

        if band_indices is None:
            band_indices = list(range(1, count + 1))  # 1-based

        if titles is None:
            titles = [f"Band {i}" for i in band_indices]

        num_bands = len(band_indices)
        ncols = 3
        nrows = int(np.ceil(num_bands / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
        axes = np.array(axes).flatten()

        for i, idx in enumerate(band_indices):
            band = src.read(idx)
            band = np.where(band == src.nodata, np.nan, band)


            im = axes[i].imshow(band, cmap=cmap)
            axes[i].set_title(titles[i], fontsize=12)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], shrink=0.6)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()



'''
remove_bands_from_raster(
    input_raster_path="data/features/uttarakhand_full_stack_2024-06-01.tif",
    output_raster_path="data/features/uttarakhand_full_stack_2024-06-01.tif",
    bands_to_remove=[1,2,3]

)
'''

def delete_previous_data(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

#display_multiband_raster("data/features/uttarakhand_full_stack_2024-06-01.tif")

#ONLY GENERATE FIRE DATA FOR THOSE DAYS WITH FIRE!!!
class FireCastDataGenerator():

    def __init__(self,date):
        self.date2 = date
        # Convert to datetime object
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        next_day = date_obj - timedelta(days=1)
        self.date1 = next_day.strftime("%Y-%m-%d")
        delete_previous_data("data/features")
        delete_previous_data("data/targets")
        delete_previous_data("data/weather")
        delete_previous_data("data/temp")


    
    def stack_weather_with_static(self, static_raster_path, weather_raster_path, output_path):
        # === Open static raster (e.g., slope, aspect, LULC, GHS)
        with rasterio.open(static_raster_path) as static_src:
            static_data = static_src.read()  # shape: (bands, H, W)
            static_meta = static_src.meta.copy()

        # === Open weather raster (should already be preprocessed to match)
        with rasterio.open(weather_raster_path) as weather_src:
            weather_data = weather_src.read()  # shape: (bands, H, W)

            # Safety check: ensure same shape
            if static_data.shape[1:] != weather_data.shape[1:]:
                raise ValueError("❌ Shape mismatch between static and weather rasters.")

            if static_src.crs != weather_src.crs:
                raise ValueError("❌ CRS mismatch. Please preprocess weather raster to match static raster.")

        # === Stack along band axis
        combined_stack = np.vstack([static_data, weather_data])  # shape: (all_bands, H, W)

        # === Update metadata
        out_meta = static_meta.copy()
        out_meta.update({
            'count': combined_stack.shape[0],
            'dtype': 'float32'
        })

        # === Save combined raster
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(combined_stack.astype(np.float32))

        print(f"✅ Stacked raster saved to: {output_path}")

    def generate_weather_data(self):
        load_weather.weather_data(self.date1)
        path = load_weather.extract("era5_sample.nc")
        path = load_weather.convert_nc_to_multiband_tif(path,'data/weather')
        load_weather.preprocess_and_normalize_multiband_raster(
            input_tif_path=path,
            boundary_shapefile_path="data/boundary_files/uttarakhand_boundary_32644.shp",
            reference_raster_path="uttarakhand_static_stack.tif",
            output_path=path
        )
        self.stack_weather_with_static(
            static_raster_path="uttarakhand_static_stack.tif",
            weather_raster_path=path,
            output_path=f"data/features/uttarakhand_full_stack_{self.date1}.tif"
        )

    def generate_fire_data(self):

        load_fire.clip_and_rasterize_normalized_frp(
            fire_shapefile_path="data/fire_data/fire_nrt_SV-C2_630938.shp",
            boundary_shapefile_path="data/boundary_files/uttarakhand_boundary_32644.shp",
            reference_raster_path="uttarakhand_static_stack.tif",
            output_raster_path=f"data/targets/fire_target_{self.date2}.tif",
            target_date=self.date2
        )

l1,l2 = load_fire.find_dates_with_fire("data/fire_data/fire_nrt_SV-C2_630938.shp","data/boundary_files/uttarakhand_boundary_32644.shp")
#93 fire data, make it a 1:3 fire to no fire ratio

print(l1[2])
'''
gen = FireCastDataGenerator(l1[2])

gen.generate_weather_data()
gen.generate_fire_data()
'''


model = FireCastModel()
model.train(l1[2])