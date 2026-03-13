# 🔥 FireFiX - AI-Powered Fire Detection System

FireFiX is an intelligent fire detection system that combines deep learning, computer vision, and sensor data to detect fires in real-time with high accuracy.

## 📋 Features

- **Deep Learning Model**: MobileNetV2-based CNN trained for fire classification
- **GradCAM Visualization**: Visual explanation of model predictions with attention heatmaps
- **Multi-Modal Detection**: Combines CNN predictions, flame color detection, and sensor data (temperature, smoke, humidity)
- **Real-Time API**: Flask-based server for processing live camera feeds
- **Visual Dashboard**: Real-time visualization with fire probability, sensor readings, and confidence bars
- **Bounding Box Detection**: Identifies specific fire regions in frames

## 🛠️ Requirements

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster training)
- Webcam or camera module (for real-time detection)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd FireFiX
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch
- torchvision
- Flask
- OpenCV (cv2)
- numpy
- Pillow
- pytorch-grad-cam

## 🚀 Quick Start

### Step 1: Prepare Your Dataset

If you have XML annotations (Pascal VOC format):

```bash
python convert_dataset.py
```

This will organize your images into `fire_dataset/fire/` and `fire_dataset/nofire/` directories.

**Note:** Update the paths in `convert_dataset.py` before running:
```python
images_path = "Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample"
annotations_path = "Annotations/Annotations"
```

### Step 2: Train the Model

```bash
python train_fire_model.py
```

This will:
- Train a MobileNetV2 model on your dataset
- Use transfer learning with ImageNet pretrained weights
- Save the trained model as `fire_model.pth`

**Training parameters:**
- Batch size: 16
- Epochs: 10
- Learning rate: 0.001
- Optimizer: Adam

### Step 3: Test the Model

Test on a single image:

```bash
python test_model.py
```

**Note:** Change the image path in `test_model.py`:
```python
img_path = "test_fire.jpg"  # Update this to your test image
```

### Step 4: Visualize with GradCAM

Generate attention heatmaps to see what the model is focusing on:

```bash
python gradcam_test.py
```

**Note:** Update the image path in `gradcam_test.py` before running.

### Step 5: Run the Detection Server

Start the Flask API server:

```bash
python server.py
```

The server will start on `http://0.0.0.0:5000`

### Step 6: Simulate Detection with an MP4 Video

If you want to test the model without real sensor hardware, you can run the same detection pipeline on a fire video and inject simulated sensor readings.

```bash
python server.py --video path\to\house_fire.mp4 --temperature 58 --humidity 35 --smoke 2800
```

This runs in a separate offline mode and does not change the normal `/upload` behavior used with live sensor values.
GradCAM is disabled by default in video mode so playback is usable on CPU. If you want the heatmap anyway, add `--with-gradcam`.

You can also vary the sensor values over time with comma-separated sequences:

```bash
python server.py --video path\to\house_fire.mp4 --temperature-seq 28,30,34,45,58 --smoke-seq 900,1200,1700,2400,3200
```

Optional flags:
- `--frame-stride 2` processes every second frame.
- `--loop` repeats the video until you press `q`.
- `--with-gradcam` enables GradCAM during video simulation, but it is much slower.

## 📡 API Usage

### Endpoint: `/upload`

Send POST requests with image data and sensor readings.

**Request:**
- **Method:** POST
- **Content-Type:** `image/jpeg` (raw binary data)
- **Headers:**
  - `Temperature`: Temperature value (°C)
  - `Humidity`: Humidity percentage
  - `Smoke`: Smoke sensor reading

**Example using Python requests:**

```python
import requests
import cv2

# Capture frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
_, img_encoded = cv2.imencode('.jpg', frame)

# Send to server
headers = {
    'Temperature': '25.5',
    'Humidity': '60',
    'Smoke': '1200'
}

response = requests.post(
    'http://localhost:5000/upload',
    data=img_encoded.tobytes(),
    headers=headers
)

print(response.text)  # "FIRE" or "SAFE"
```

**Response:**
- `FIRE` - Fire detected (200 OK)
- `SAFE` - No fire detected (200 OK)
- `ERROR` - Processing error (400/500)

## 🗂️ Project Structure

```
FireFiX/
├── convert_dataset.py           # Convert XML annotations to fire/nofire dataset
├── train_fire_model.py          # Train the MobileNetV2 fire detection model
├── test_model.py                # Test model on single images
├── gradcam_test.py              # Generate GradCAM visualizations
├── server.py                    # Flask API server for real-time detection
├── fire_model.pth               # Trained model weights
├── Annotations/                 # XML annotation files (Pascal VOC format)
│   └── Annotations/
├── Datacluster Fire and Smoke Sample/  # Source images
│   └── Datacluster Fire and Smoke Sample/
└── fire_dataset/                # Organized training dataset
    ├── fire/                    # Fire images
    └── nofire/                  # Non-fire images
```

## 🔬 How It Works

### 1. Dataset Preparation
The `convert_dataset.py` script parses XML annotations to identify images containing fire and organizes them into a classification dataset.

### 2. Model Training
- **Architecture:** MobileNetV2 (lightweight and efficient)
- **Transfer Learning:** Uses ImageNet pretrained weights
- **Fine-tuning:** Only the classifier layer is trained initially
- **Data Augmentation:** Random flips, rotations, and color jittering

### 3. Fire Detection Logic

The system uses multiple detection methods for robustness:

#### CNN Detection
- Probability threshold: 85%
- Uses temporal smoothing (5-frame buffer) to reduce false positives

#### Flame Color Detection
- HSV color space filtering for yellow/orange flames
- Flame ratio threshold: 5% of frame

#### Sensor Thresholds
- Temperature: > 50°C
- Smoke: > 2500 units

**Fire is detected if:**
- (CNN detects fire AND flame color detected) OR
- Smoke threshold exceeded OR
- Temperature threshold exceeded

### 4. Visual Dashboard

The server provides a 2x2 grid display:
- **Top Left:** Original frame with bounding boxes and status
- **Top Right:** GradCAM heatmap overlay
- **Bottom Left:** Flame mask visualization
- **Bottom Right:** Processed frame

Includes:
- Fire probability bar
- Real-time FPS counter
- Sensor readings
- Fire confidence metric

## ⚙️ Configuration

### Modify Detection Thresholds

In `server.py`, adjust these values:

```python
# CNN threshold
cnn_fire = fire_prob > 0.85  # Default: 0.85

# Flame color threshold
flame_fire = flame_ratio > 0.05  # Default: 0.05

# Sensor thresholds
smoke_fire = smoke > 2500  # Default: 2500
temp_fire = temp > 50  # Default: 50°C
```

### Adjust Training Parameters

In `train_fire_model.py`:

```python
epochs = 10  # Number of training epochs
batch_size = 16  # Batch size
lr = 0.001  # Learning rate
```

## 🎯 Performance Tips

1. **GPU Acceleration:** Install CUDA-enabled PyTorch for faster training
2. **Model Optimization:** Use TorchScript or ONNX for production deployment
3. **Dataset Balance:** Ensure roughly equal fire/nofire samples
4. **Data Augmentation:** More augmentation helps with diverse environments
5. **Temporal Smoothing:** Increase buffer size for more stable predictions

## 🐛 Troubleshooting

### Model not loading
- Ensure `fire_model.pth` exists in the project root
- Check that the model architecture matches the saved weights

### Poor detection accuracy
- Train longer (increase epochs)
- Add more training data
- Adjust detection thresholds
- Check data quality and labeling

### Server not displaying dashboard
- Ensure OpenCV GUI is available (not in headless environment)
- Check that the camera/image data is valid

### Import errors
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

## 📝 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Made with ❤️ for fire safety and prevention**
