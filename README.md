# FireCast.AI
"Forecasting and simulating forest fire spread using AI/ML."

Simulation/Modelling of Forest Fire Spread using AI/ML techniques
Uncontrolled forest fires represent a significant challenge for government agencies tasked with preserving biodiversity and maintaining air quality. The spread of such fires is influenced by factors including weather conditions (temperature, precipitation, humidity, wind), terrain (slope, aspect, fuel availability), and human activity. With modern geospatial technologies, datasets from the Forest Survey of India and global services like VIIRS-SNP are accessible. Despite this, real-time simulation and forecasting remain complex. Short-term forecasting and dynamic simulation are crucial for timely preventive measures. AI/ML techniques offer promising capabilities to extract insights, helping planners estimate damage, prioritize containment, and mitigate fire impacts.

a) Fire prediction map for the next day. (SPATIAL MODEL)
b) Simulated fire spread with animation for 1/2/3/6/12 hours. (TEMPORAL MODEL)

Models to USE:

🔥 1) Spatial Fire Prediction Model
Goal: Predict a fire risk map for the next day from today's inputs (weather, LULC, slope, etc.)

✅ Recommended Model: U-Net or U-Net++
Type: Fully Convolutional Neural Network (CNN)

Input: Multi-channel raster (e.g., [128×128×C])

Output: Same-sized raster with per-pixel fire class or probability

📌 Why U-Net?
Excellent for image-to-image tasks

Captures spatial patterns and context (like slope + vegetation + wind)

Works well even with limited data via data augmentation

Can predict multi-class labels: nil / low / moderate / high

Optional Enhancements:
U-Net++ (nested skip connections) for deeper feature learning

Use softmax output for class probabilities

🔥 2) Temporal Fire Spread Simulation Model
Goal: Simulate and animate fire spread across 1–12 hours from a starting fire map, factoring in wind, slope, fuel, etc.

✅ Option A: Cellular Automata (CA) + Rule-based model
Type: Discrete simulation model

Core idea: Each pixel (cell) updates its state based on neighboring cells, slope, wind direction, and fuel availability

Highly interpretable and intuitive

🔧 Components:
Fire rules (e.g., if neighboring cell is burning + wind in that direction → ignite)

Speed modifiers from DEM slope, aspect, fuel

Simulate over time with time-steps (1h, 2h, 3h, …)

Easy to animate using matplotlib or GIS tools

✅ Option B: ConvLSTM (Convolutional LSTM)
Type: Spatio-temporal sequence-to-sequence model

Input: Sequence of past fire maps + wind/slope/fuel layers

Output: Sequence of predicted fire spread maps (e.g., for each hour)

📌 Why ConvLSTM?
Captures both temporal dynamics (fire progression) and spatial context

More accurate if enough historical fire spread data is available

Can be trained to learn spread dynamics from real cases (not just handcrafted rules)

🚀 Which to Use When?
Model	Use Case	Pros	Cons
U-Net	Predict fire for next day	Accurate, simple, interpretable	No time modeling
CA	Simulate spread over hours	Easy to animate, rule-based	Not data-learned
ConvLSTM	Predict hourly fire maps	Learns real fire behavior	Needs more data & tuning

🧰 Tools & Frameworks:
Framework	Use For	Libraries
TensorFlow / PyTorch	U-Net, ConvLSTM	tf.keras, torch, segmentation_models
QGIS / Rasterio	Data prep & visualization	rasterio, GDAL, matplotlib
Python + NumPy	Cellular Automata logic	Custom simulation & animation
