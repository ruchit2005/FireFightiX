from flask import Flask, request
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from collections import deque

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)

# Device
device = torch.device("cpu")

# -------------------------------
# Load CNN Model
# -------------------------------

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("fire_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# GradCAM setup
# -------------------------------

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# -------------------------------
# Image preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------------
# Prediction smoothing buffer
# -------------------------------

prob_buffer = deque(maxlen=5)

# -------------------------------
# Upload Endpoint
# -------------------------------

@app.route('/upload', methods=['POST'])
def upload():

    try:

        if not request.data:
            return "ERROR", 400

        # Decode incoming image
        npimg = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return "ERROR",400

        # -------------------------------
        # Prepare image for CNN
        # -------------------------------

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        tensor: torch.Tensor = transform(img_pil)  # type: ignore
        input_tensor = tensor.unsqueeze(0)

        # -------------------------------
        # CNN Prediction
        # -------------------------------

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)

        fire_prob = probs[0][1].item()

        # -------------------------------
        # Smooth predictions
        # -------------------------------

        prob_buffer.append(fire_prob)
        fire_prob = sum(prob_buffer)/len(prob_buffer)

        # -------------------------------
        # GradCAM heatmap
        # -------------------------------

        grayscale_cam = cam(input_tensor=input_tensor)[0]
        grayscale_cam = cv2.resize(grayscale_cam, (224,224))

        img_np = np.array(img_pil)/255.0

        # ensure same size as CAM
        img_np = cv2.resize(img_np, (224,224))

        heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        heatmap = cv2.resize(heatmap,(640,480))

        # -------------------------------
        # Decision Logic
        # -------------------------------

        if fire_prob > 0.85:
            status = "🔥 FIRE DETECTED"
            color = (0,0,255)
            response = "FIRE"
        else:
            status = "SAFE"
            color = (0,255,0)
            response = "SAFE"

        # -------------------------------
        # Display result
        # -------------------------------

        cv2.putText(
            heatmap,
            f"{status} ({fire_prob:.2f})",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow("Fog AI Fire Detection", heatmap)
        cv2.waitKey(1)

        print(
            f"FireProb={probs[0][1].item():.3f}",
            f"NoFireProb={probs[0][0].item():.3f}",
            f"Smoothed={fire_prob:.3f}"
        )

        return response,200

    except Exception as e:

        print("Server Error:", e)
        return "ERROR",500


# -------------------------------
# Run server
# -------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)