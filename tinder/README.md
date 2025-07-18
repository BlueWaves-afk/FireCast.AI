TINDER
Temporal Ignition and Dispersion Engine for Risk modeling

Is a Spatiotemporal model that generates a a firemap over the next 12 hour period

THE MODEL ARCHIECTURE THAT WOULD BE USED:
Option A: Cellular Automata-Based Fire Spread
-GRID BASED: each raster pixel is a cell in a grid
-Control factor: Slope, wind direction, wind speed, LULC (fuel), dryness (dew temp)

Option B: Neural Network or RNN Temporal Model

-Train a model (e.g., UNet++, ConvLSTM, or Temporal CNN) that takes:

Inputs: feature stack + predicted fire map at t

Target: fire occurrence at t+1, t+2, etc.
-Pros: Data-driven, learns implicit dynamics
-Cons: Needs large time-sequenced fire data, VIIRS/MODIS across hours/days