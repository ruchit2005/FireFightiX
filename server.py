import argparse
import ctypes
from typing import cast

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
dashboard_window_initialized = False
dashboard_window_fullscreen = False


def reset_runtime_state():
    global prev_time

    prob_buffer.clear()
    prev_time = time.time()


def _clamp_confidence(value):
    return max(0.0, min(1.0, float(value)))


def _threshold_confidence(value, threshold, warning_ratio=0.7):
    warning_level = threshold * warning_ratio
    if value <= warning_level:
        return 0.0
    if value >= threshold:
        return 1.0

    return _clamp_confidence((value - warning_level) / (threshold - warning_level))


def draw_panel(frame, top_left, bottom_right, fill_color, border_color, alpha=0.55, border_thickness=2):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, border_color, border_thickness)


def draw_text(frame, text, origin, color, scale=0.55, thickness=1):
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_metric_line(frame, label, value, origin, label_color=(180, 180, 180), value_color=(255, 255, 255)):
    x, y = origin
    draw_text(frame, label, (x, y), label_color, scale=0.48, thickness=1)
    draw_text(frame, value, (x + 88, y), value_color, scale=0.58, thickness=2)


def add_section_title(frame, title):
    draw_panel(frame, (8, 8), (118, 34), (18, 18, 18), (70, 70, 70), alpha=0.45, border_thickness=1)
    draw_text(frame, title, (18, 27), (245, 245, 245), scale=0.48, thickness=1)


def draw_confidence_bar(frame, label, value, origin, size, fill_color, border_color, highlighted=False):
    x, y = origin
    width, height = size
    value = _clamp_confidence(value)

    thickness = 3 if highlighted else 2
    fill_width = int((width - 4) * value)

    draw_text(frame, label, (x, y - 8), border_color, scale=0.42, thickness=1)

    cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, thickness)
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), fill_color, -1)
    draw_text(frame, f"{value:.2f}", (x + width + 8, y + height - 1), border_color, scale=0.42, thickness=1)


def draw_info_panel(frame, status, status_color, final_confidence, temp, smoke, humidity, fps, cnn_confidence):
    draw_panel(frame, (10, 12), (146, 152), (16, 18, 24), (60, 60, 60), alpha=0.62)
    draw_text(frame, status, (20, 38), status_color, scale=0.75, thickness=2)
    draw_text(frame, f"Final {final_confidence:.2f}", (20, 60), (255, 215, 0), scale=0.5, thickness=1)

    draw_metric_line(frame, "Temp", f"{temp:.1f} C", (20, 86))
    draw_metric_line(frame, "Smoke", str(smoke), (20, 106))
    draw_metric_line(frame, "Humidity", f"{humidity:.1f}%", (20, 126))
    draw_metric_line(frame, "FPS", f"{fps:.1f}", (20, 146))
    draw_text(frame, f"CNN {cnn_confidence:.2f}", (20, 170), (215, 230, 255), scale=0.48, thickness=1)


def draw_corner_panel(frame, final_confidence, cnn_confidence, sensor_confidence):
    x1, y1, x2, y2 = 168, 12, 308, 114
    draw_panel(frame, (x1, y1), (x2, y2), (14, 18, 22), (90, 90, 90), alpha=0.68)
    draw_text(frame, "CONFIDENCE", (178, 32), (240, 240, 240), scale=0.48, thickness=1)

    draw_confidence_bar(
        frame,
        "FINAL",
        final_confidence,
        (178, 48),
        (92, 12),
        (0, 215, 255),
        (0, 215, 255),
        highlighted=True,
    )
    draw_confidence_bar(
        frame,
        "CNN",
        cnn_confidence,
        (178, 74),
        (92, 10),
        (0, 140, 255),
        (230, 230, 230),
    )
    draw_confidence_bar(
        frame,
        "SENSOR",
        sensor_confidence,
        (178, 97),
        (92, 10),
        (0, 200, 0),
        (230, 230, 230),
    )


def draw_bottom_bar(frame, final_confidence):
    draw_panel(frame, (10, 186), (310, 232), (18, 18, 18), (70, 70, 70), alpha=0.62)
    draw_confidence_bar(
        frame,
        "FINAL FIRE CONFIDENCE",
        final_confidence,
        (20, 208),
        (250, 14),
        (0, 215, 255),
        (255, 215, 0),
        highlighted=True,
    )


def get_screen_size(default_width, default_height):
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return int(default_width), int(default_height)


def process_frame(frame, temp=0.0, humidity=0.0, smoke=0, enable_gradcam=True):
    global prev_time

    if frame is None:
        raise ValueError("Frame is empty")

    # ---------------- CNN ----------------

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((224,224))

    tensor = cast(torch.Tensor, transform(img_pil)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    fire_prob = probs[0][1].item()

    prob_buffer.append(fire_prob)
    fire_prob = sum(prob_buffer)/len(prob_buffer)

    # ---------------- GradCAM ----------------

    if enable_gradcam:
        grayscale_cam = cam(input_tensor=tensor)[0]
        grayscale_cam = cv2.resize(grayscale_cam,(224,224))

        img_np = np.array(img_pil)/255.0
        heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        heatmap = cv2.resize(heatmap,(320,240))
    else:
        heatmap = np.zeros((240,320,3), dtype=np.uint8)
        cv2.putText(heatmap,"GradCAM disabled for video test",(12,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

    # ---------------- Flame Mask ----------------

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0,150,200])
    upper = np.array([25,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    flame_ratio = np.sum(mask > 0) / mask.size

    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    mask_color = cv2.resize(mask_color,(320,240))

    # ---------------- Bounding Box ----------------

    display_frame = frame.copy()
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(display_frame,(x,y),(x+w,y+h),(0,0,255),2)

    # ---------------- Fire Decision ----------------

    cnn_fire = fire_prob > 0.85
    flame_fire = flame_ratio > 0.05
    smoke_fire = smoke > 2500
    temp_fire = temp > 50
    flame_confidence = _clamp_confidence(flame_ratio / 0.05)
    temp_confidence = _threshold_confidence(temp, 50.0)
    smoke_confidence = _threshold_confidence(smoke, 2500.0)
    sensor_confidence = max(temp_confidence, smoke_confidence)
    vision_confidence = _clamp_confidence((fire_prob * 0.7) + (flame_confidence * 0.3))
    final_confidence = max(vision_confidence, sensor_confidence)

    if (cnn_fire and flame_fire) or smoke_fire or temp_fire:
        status = "FIRE DETECTED"
        color = (0,0,255)
        response = "FIRE"
    else:
        status = "SAFE"
        color = (0,255,0)
        response = "SAFE"

    display_frame = cv2.resize(display_frame,(320,240))

    # ---------------- FPS ----------------

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    # ---------------- Text Overlay ----------------

    draw_info_panel(display_frame, status, color, final_confidence, temp, smoke, humidity, fps, fire_prob)

    # ---------------- Confidence Bar ----------------

    draw_corner_panel(display_frame, final_confidence, fire_prob, sensor_confidence)
    draw_bottom_bar(display_frame, final_confidence)

    add_section_title(heatmap, "GRADCAM")
    add_section_title(mask_color, "FLAME MASK")

    diagnostics_frame = display_frame.copy()
    add_section_title(display_frame, "LIVE VIEW")
    add_section_title(diagnostics_frame, "DIAGNOSTICS")

    # ---------------- Dashboard ----------------

    top = np.hstack((display_frame, heatmap))
    bottom = np.hstack((mask_color, diagnostics_frame))

    dashboard = np.vstack((top,bottom))

    return {
        "response": response,
        "status": status,
        "fire_prob": fire_prob,
        "sensor_confidence": sensor_confidence,
        "final_confidence": final_confidence,
        "flame_ratio": flame_ratio,
        "fps": fps,
        "dashboard": dashboard,
    }


def show_dashboard(dashboard, wait_ms=1, window_name="AI Fire Detection Dashboard", fullscreen=False):
    global dashboard_window_initialized, dashboard_window_fullscreen

    if not dashboard_window_initialized:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        dashboard_window_initialized = True

    if fullscreen and not dashboard_window_fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        dashboard_window_fullscreen = True

    if fullscreen:
        target_width, target_height = get_screen_size(dashboard.shape[1], dashboard.shape[0])
        dashboard_to_show = cv2.resize(dashboard, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        dashboard_to_show = dashboard

    cv2.imshow(window_name, dashboard_to_show)
    return cv2.waitKey(wait_ms) & 0xFF


def _resolve_sensor_value(sensor_values, key, frame_index, default_value):
    if sensor_values is None:
        return default_value

    value = sensor_values.get(key, default_value)

    if isinstance(value, (list, tuple)):
        if not value:
            return default_value
        return value[min(frame_index, len(value) - 1)]

    return value


def _parse_sequence(raw_value, cast_type):
    if not raw_value:
        return None

    return [cast_type(item.strip()) for item in raw_value.split(",") if item.strip()]


def test_video_with_simulated_sensors(video_path, sensor_values=None, frame_stride=1, loop=False, enable_gradcam=False, fullscreen=False):
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    reset_runtime_state()
    frame_index = 0
    processed_frames = 0

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = max(1, int(1000 / source_fps)) if source_fps and source_fps > 0 else 33

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            temp = float(_resolve_sensor_value(sensor_values, "temperature", frame_index, 0.0))
            humidity = float(_resolve_sensor_value(sensor_values, "humidity", frame_index, 0.0))
            smoke = int(_resolve_sensor_value(sensor_values, "smoke", frame_index, 0))

            result = process_frame(
                frame,
                temp=temp,
                humidity=humidity,
                smoke=smoke,
                enable_gradcam=enable_gradcam,
            )
            pressed_key = show_dashboard(result["dashboard"], wait_ms=wait_ms, fullscreen=fullscreen)

            processed_frames += 1
            frame_index += 1

            if pressed_key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return processed_frames


def build_sensor_values(args):
    sensor_values = {
        "temperature": args.temperature,
        "humidity": args.humidity,
        "smoke": args.smoke,
    }

    temperature_sequence = _parse_sequence(args.temperature_seq, float)
    humidity_sequence = _parse_sequence(args.humidity_seq, float)
    smoke_sequence = _parse_sequence(args.smoke_seq, int)

    if temperature_sequence is not None:
        sensor_values["temperature"] = temperature_sequence

    if humidity_sequence is not None:
        sensor_values["humidity"] = humidity_sequence

    if smoke_sequence is not None:
        sensor_values["smoke"] = smoke_sequence

    return sensor_values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Run offline simulation from an MP4 or other video file")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default temperature value for video simulation")
    parser.add_argument("--humidity", type=float, default=0.0, help="Default humidity value for video simulation")
    parser.add_argument("--smoke", type=int, default=0, help="Default smoke value for video simulation")
    parser.add_argument("--temperature-seq", help="Comma-separated temperature values applied across frames")
    parser.add_argument("--humidity-seq", help="Comma-separated humidity values applied across frames")
    parser.add_argument("--smoke-seq", help="Comma-separated smoke values applied across frames")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame during video simulation")
    parser.add_argument("--loop", action="store_true", help="Loop the video until you press q")
    parser.add_argument("--with-gradcam", action="store_true", help="Enable GradCAM in video simulation mode. This is much slower on CPU.")
    parser.add_argument("--fullscreen", action="store_true", help="Show dashboard in fullscreen and stretch panels to cover the entire screen")
    return parser.parse_args()

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

        result = process_frame(frame, temp=temp, humidity=humidity, smoke=smoke)
        show_dashboard(result["dashboard"])

        return result["response"],200

    except Exception as e:
        print("Error:",e)
        return "ERROR",500


if __name__ == "__main__":
    args = parse_args()

    if args.video:
        processed_frames = test_video_with_simulated_sensors(
            args.video,
            sensor_values=build_sensor_values(args),
            frame_stride=args.frame_stride,
            loop=args.loop,
            enable_gradcam=args.with_gradcam,
            fullscreen=args.fullscreen,
        )
        print(f"Processed {processed_frames} frames from {args.video}")
    else:
        app.run(host="0.0.0.0",port=5000,threaded=False)