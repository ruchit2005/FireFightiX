from flask import Flask, request
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from collections import deque
import time

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)

device = torch.device("cpu")

# -----------------------------
# Load CNN
# -----------------------------

model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("fire_model.pth", map_location=device))
model.eval()

# -----------------------------
# GradCAM
# -----------------------------

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

prob_buffer = deque(maxlen=5)

prev_time = time.time()

# -----------------------------
# Upload Endpoint
# -----------------------------

@app.route('/upload', methods=['POST'])
def upload():

    global prev_time

    try:

        temp = float(request.headers.get("Temperature",0))
        humidity = float(request.headers.get("Humidity",0))
        smoke = int(request.headers.get("Smoke",0))

        npimg = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return "ERROR",400

        # ---------------- CNN ----------------

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((224,224))

        tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)

        fire_prob = probs[0][1].item()

        prob_buffer.append(fire_prob)
        fire_prob = sum(prob_buffer)/len(prob_buffer)

        # ---------------- GradCAM ----------------

        grayscale_cam = cam(input_tensor=tensor)[0]
        grayscale_cam = cv2.resize(grayscale_cam,(224,224))

        img_np = np.array(img_pil)/255.0
        heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        heatmap = cv2.resize(heatmap,(320,240))

        # ---------------- Flame Mask ----------------

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0,150,200])
        upper = np.array([25,255,255])

        mask = cv2.inRange(hsv, lower, upper)

        flame_ratio = np.sum(mask > 0) / mask.size

        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        mask_color = cv2.resize(mask_color,(320,240))

        # ---------------- Bounding Box ----------------

        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # ---------------- Fire Decision ----------------

        cnn_fire = fire_prob > 0.85
        flame_fire = flame_ratio > 0.05
        smoke_fire = smoke > 2500
        temp_fire = temp > 50

        if (cnn_fire and flame_fire) or smoke_fire or temp_fire:
            status = "🔥 FIRE DETECTED"
            color = (0,0,255)
            response = "FIRE"
        else:
            status = "SAFE"
            color = (0,255,0)
            response = "SAFE"

        frame = cv2.resize(frame,(320,240))

        # ---------------- FPS ----------------

        current_time = time.time()
        fps = 1/(current_time-prev_time)
        prev_time = current_time

        # ---------------- Text Overlay ----------------

        cv2.putText(frame,f"{status} ({fire_prob:.2f})",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        cv2.putText(frame,f"TEMP: {temp:.1f}C",(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"SMOKE: {smoke}",(10,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"FPS: {fps:.1f}",(10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # ---------------- Confidence Bar ----------------

        bar_x = int(fire_prob * 300)

        cv2.rectangle(frame,(10,200),(310,220),(100,100,100),2)
        cv2.rectangle(frame,(10,200),(10+bar_x,220),(0,0,255),-1)

        cv2.putText(frame,"FIRE CONFIDENCE",(10,195),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # ---------------- Dashboard ----------------

        top = np.hstack((frame, heatmap))
        bottom = np.hstack((mask_color, frame))

        dashboard = np.vstack((top,bottom))

        cv2.imshow("AI Fire Detection Dashboard",dashboard)
        cv2.waitKey(1)

        return response,200

    except Exception as e:
        print("Error:",e)
        return "ERROR",500


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,threaded=False)