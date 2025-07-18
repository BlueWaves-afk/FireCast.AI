
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scripts.load_weather as load_weather
import scripts.load_fire as load_fire
import shutil
from prism.model import FireCastPRISMModel
import pickle
import logging 
import matplotlib.colors as mcolors
from collections import Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

        logger.info(f"📦 Original Bands: {all_indices}")
        logger.info(f"❌ Removing Bands: {bands_to_remove}")
        logger.info(f"✅ Keeping Bands: {keep_indices}")

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

    logger.info(f"✅ New raster saved to: {output_raster_path}")

def display_multiband_raster(raster_path, band_indices=None, titles=None, cmap='viridis', normalize=True):
    """
    Displays selected bands from a multi-band raster.

    Parameters:
        raster_path (str): Path to the multi-band raster file.
        band_indices (list, optional): List of band indices to display (1-based). If None, displays all bands.
        titles (list, optional): Titles for each band. If None, uses default titles.
        cmap (str): Colormap to use for displaying bands.
        normalize (bool): Whether to normalize band values to [0, 1].
    """

    with rasterio.open(raster_path) as src:
        count = src.count
        logger.info(f"📊 Number of bands: {count}")

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

def plot_fire_risk_map_with_legend(raster_path, title="Fire Risk Classification Map"):
    # Load raster
    with rasterio.open(raster_path) as src:
        fire_map = src.read(1)

    # Create a masked array for nodata = 255
    fire_map_masked = np.ma.masked_where(fire_map == 255, fire_map)

    # Define custom colormap (0–4 classes + gray for nodata)
    cmap = mcolors.ListedColormap(['darkgreen', 'green', 'yellow', 'orange', 'red', 'gray'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 255.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Extend fire_map to mark 255 values explicitly
    fire_map_vis = np.where(fire_map == 255, 5, fire_map)  # class index 5 = NoData/gray

    # Define labels
    class_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'No Data']

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(fire_map_vis, cmap=cmap, norm=norm)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(class_labels)
    cbar.set_label("Fire Risk Class", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def delete_previous_data(path):
    """
    Deletes all files and directories in the specified path.
    Parameters:
        path (str): Path to the directory to clean.
    """

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

#display_multiband_raster("data/features/uttarakhand_full_stack_2024-06-01.tif")

def compute_class_frequencies_from_dates(date_list, generator_class, tmp_tif_path, valid_classes=[0,1,2,3,4], ignore_value=255):
    """
    Args:
        date_list: List of date strings (e.g., ['2024-01-01', '2024-01-02', ...])
        generator_class: Your data generator class (e.g., FireCastDataGenerator)
        tmp_tif_path: Temporary path where raster will be saved (e.g., "temp_target.tif")
    """
    total_pixels = 0
    class_counts = Counter()

    for date in date_list:
        print(f"📅 Processing: {date}")
        gen = generator_class(date)
        gen.generate_fire_data()  # This should save the raster to tmp_tif_path

        if not os.path.exists(tmp_tif_path):
            print(f"❌ TIF not found: {tmp_tif_path}")
            continue

        # Read the generated raster
        with rasterio.open(tmp_tif_path) as src:
            data = src.read(1)
            mask = (data != ignore_value)
            valid_data = data[mask]

            # Count occurrences
            unique, counts = np.unique(valid_data, return_counts=True)
            class_counts.update(dict(zip(unique, counts)))
            total_pixels += valid_data.size

        # Delete raster after processing
        os.remove(tmp_tif_path)

    if total_pixels == 0:
        raise ValueError("No valid pixels found across all dates.")

    # Compute normalized class frequencies
    frequencies = {cls: class_counts.get(cls, 0) / total_pixels for cls in valid_classes}

    print("📊 Class Frequencies across all dates:")
    for cls, freq in frequencies.items():
        print(f"  Class {cls}: {freq:.6f}")

    return frequencies


class FireCastDataGenerator():

    def __init__(self,date):
        """
        Initializes the data generator with a specific date.
        The date should be in the format "YYYY-MM-DD".
        """
        self.date = date


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
        delete_previous_data(os.path.dirname(output_path))  # Clean previous data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(combined_stack.astype(np.float32))

        logger.info(f"✅ Stacked raster saved to: {output_path}")

    def generate_weather_data(self):

        load_weather.weather_data(self.date)
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
            output_path=f"data/features/uttarakhand_full_stack.tif"
        )

        logger.info("✅ Weather data generation complete.")

    def generate_fire_data(self):

        load_fire.clip_and_rasterize_normalized_frp(
            fire_shapefile_path="data/fire_data/fire_nrt_SV-C2_630938.shp",
            boundary_shapefile_path="data/boundary_files/uttarakhand_boundary_32644.shp",
            reference_raster_path="uttarakhand_static_stack.tif",
            output_raster_path=f"data/targets/fire_target.tif",
            target_date=self.date
        )

        logger.info("✅ Fire data generation complete.")




if __name__ == "__main__":


    
    while True:
        # Load current queue
        try:
            with open("date_queue.pkl", "rb") as f:
                date_list = pickle.load(f)
        except FileNotFoundError:
            logger.error("❌ date_queue.pkl not found.")
            break

        # Exit if no dates left
        if not date_list:
            logger.info("✅ All dates processed!")
            break

        # Get next date to process
        current_day = date_list[0]
        logger.info(f"📅 Processing: {current_day}")

        try:
            # === Data Generation ===
            gen = FireCastDataGenerator(current_day)
            gen.generate_fire_data()
            gen.generate_weather_data()
            

            # === Model Training ===
            model = FireCastPRISMModel()
            model.train()

            # === Optional Prediction / Evaluation ===
            # model.predict(raster_path="data/features/uttarakhand_full_stack.tif")
            # plot_fire_risk_map_with_legend(
            #     raster_path="data/targets/fire_target.tif",
            #     title=f"Fire Risk Target Map - {current_day}"
            # )

        except Exception as e:
            logger.exception(f"⚠️ Error processing {current_day}. Skipping to next date. error{e}")
        finally:
            # Always pop the date, even on error
            date_list.pop(0)

            # Save updated queue
            with open("date_queue.pkl", "wb") as f:
                pickle.dump(date_list, f)


