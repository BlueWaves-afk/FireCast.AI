🔥 PRISM — Prediction of Risk using Integrated Spatial Maps

PRISM is a geospatial deep learning system that predicts forest fire risk using the U-Net++ architecture. It generates multiclass probability maps to classify fire-prone regions from multi-source raster datasets including terrain, weather, and vegetation indicators.

🚀 Overview PRISM employs U-Net++, a powerful CNN-based segmentation model, enhanced with nested skip connections to enable rich spatial learning — especially important when dealing with complex, high-resolution geospatial data.

The system is built to operate on multi-channel raster inputs, and outputs pixel-level fire risk classification across a region, making it ideal for early warning systems, planning, and mitigation.

📦 Input Data Multi-channel raster stacks are prepared from:

Land Use / Land Cover (LULC)

Elevation & Slope

Population Data

Meteorological data:

Temperature

Wind speed/direction

Relative humidity

Each raster tile is stacked as input channels, normalized, and aligned to a common projection and resolution.

🧠 Model Architecture Base model: U-Net++

Input: Multi-channel geospatial raster stack (e.g., 256×256×M)

Output: Segmentation map of shape (256×256) with integer class labels:

0: No Risk

1: Low Risk

2: Medium Risk

3: High Risk

4: Extreme Risk

255: No Data (ignored during training)

🧮 Training Details Loss Function:

Categorical Cross Entropy or Focal Loss

Loss ignores pixels labeled 255 (e.g., outside boundary/mask region)

Optimizer:

Adam optimizer with default β1/β2 parameters

Well-suited for complex spatial models with sparse gradients

Learning Rate Scheduler:

ReduceLROnPlateau or CosineAnnealingLR to adapt learning rate dynamically

Helps converge faster while avoiding overfitting

Device:

Trained on CUDA-enabled GPUs (torch.device("cuda")) for acceleration

Falls back to CPU if no GPU is available

Gradient Accumulation:

For systems with memory constraints, gradient accumulation is used to simulate larger batch sizes across multiple forward passes:

🗺️ Output Generates a multiclass probability map over the given raster tile or region. The predicted segmentation is typically post-processed into GeoTIFF or PNG format and used for visualization and further decision-making.

🌍 Applications 🔥 Real-time fire monitoring and early warning

📍 Identification of fire-prone hotspots

🚒 Disaster preparedness and resource deployment planning

📡 Integration with dashboards and GIS tools

🧪 TODOs & Enhancements Integrate temporal modeling (e.g., LSTM or ConvLSTM for fire spread)

Streamlined data preprocessing CLI

Interactive map viewer for output visualization