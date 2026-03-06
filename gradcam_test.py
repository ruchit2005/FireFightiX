import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cpu")

# Load model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("fire_model.pth", map_location=device))
model.eval()

# Last convolution layer
target_layers = [model.features[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Load test image
img_path = "test_fire.jpg"

img = Image.open(img_path).convert("RGB")
img = img.resize((224,224))
img_np = np.array(img) / 255.0
input_tensor = transform(img).unsqueeze(0)

# Generate heatmap
grayscale_cam = cam(input_tensor=input_tensor)[0]

visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

cv2.imshow("Fire GradCAM", visualization)
cv2.waitKey(0)