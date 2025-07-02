PRISM
Prediction of Risk using Integrated Spatial Maps

Is a spatial model that generates a probability map for fire risk regions using U-NET archiecture


# Model Choice
The choice of model is a very pertinent matter, there are several existing architectures like U-Net, U-Net++, and other transformer based ones. In short our model must be able to have:
1) Generalization across diverse terrain and vegetation
2) High pixel-level accuracy for fire probability classification
3) Practical training feasibility (compute + time)

For this purpose we have settled on U-NET++ architecture after extensive research, as other transformer based models, although would capture longer context, would take much processing time and gpu power.
 ------------------

 # Datasets Used
OUR DATA PIPELINE FOLLOWS:
-Clipping to state boundary
-reproject to common coordinate epsg32644
-cleaning the data for missing values
-and processing and labelling the data for ML use case

OUR COMMON RASTER DATA WOULD INCLUDE: 1)SLOPE & ASPECT 2)GHS_built_s 3) temperature 4)wind velocity 5)LULC, BUT for the purpose of generating a fire ignition model, we can avoid slope & aspect, and wind velocity as they are more usefull in predicting where the fire will spread after ignition.

We consider weather bands as dynamic, changing day to day, and keep the other bands as relatively static(update every year), hence we need to create a folder structure for seperate weather data for seperate days, across the year. We match this with VIIRS data for the next day(forecast)
We match weather and VIIRS data at 12 UTC

a) Weather Data: Wind speed/direction, temperature, rainfall, humidity (from MOSDAC, ERA-5, IMD)

We have 3 datasets to obtain our data from, out of this we have determined that ERA5(ECMWF Copernicus), is the most complete for ML modeling. For this purpose we use the cdsapi python client(Copernicus Climate Data Store (CDS)).

b) Terrain Parameters: Slope and aspect (from 30m DEM available on Bhoonidhi portal)
We obtained the data in 14 tiles, covering the region of Uttarakhand, The titles are in 30m resolution(standard).

c) Thematic Data: Fuel Availability using LULC datasets

Land Use/Land Cover (LULC)

we have options such as Sentinel or Bhuvan or ESA worldwide, we opt to use a Hybrid approach, ESA WorldCover as your base raster (for uniform 10m resolution), and overlay/correct using Bhuvan LULC classes in critical fire zones (e.g., scrub, plantation, mixed forest).

This Hybrid dataset would allow us to have ML ready inputs from ESA worldwide, as well as the comprehensive fuel detail from Bhuvan; also allowing for region specifc modeling.
 
Our Hybrid approach follows a pipeline:

🧠 Overview of the Integration 2Pipeline:
🔹 Step 1: Download Datasets
✅ ESA WorldCover as GeoTIFF: direct download

✅ Bhuvan LULC as Shapefile or raster (manual from https://bhuvan.nrsc.gov.in)

🔹 Step 2: Reproject and Rasterize Bhuvan to Match ESA
Align coordinate system, resolution, and extent

🔹 Step 3: Map Bhuvan Classes to Fuel Scores
Assign numeric fuel weights to each Bhuvan class

🔹 Step 4: Fuse Datasets
Use Bhuvan-derived fuel class where available

Use ESA fallback where Bhuvan is missing

Save final fuel availability raster for ML

d) Human Settlement data:
We obtain this in two parts:
1)from the Global Human Settlement Layer dataset (GHSL), we get Settlement density, urban extent, built-up areas
-From GHSL datasets, we opt for GHS-BUILT-S (built-up %), and GHS-SMOD(Add urban classification), for our use case
2)OpenStreetMap (OSM) Roads & Human Features, we get Stressor layers — roads, tracks, urban areas, power lines, etc.


-Pythonic way to download, download to match each tile from our slope and aspect


e)VIIRS(Historical fire data)
-again, download and pythonic method to extract