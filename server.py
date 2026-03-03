from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if request.data:
        npimg = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is not None:

            # Convert to HSV (better for fire detection)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Fire color range in HSV
            lower = np.array([0, 120, 120])
            upper = np.array([35, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            fire_ratio = np.sum(mask > 0) / mask.size
            print(f"Fire Ratio: {fire_ratio:.3f}")

            if fire_ratio > 0.30:
                status = "🔥 FIRE DETECTED"
                color = (0, 0, 255)
                response = "FIRE"
            else:
                status = "No Fire"
                color = (0, 255, 0)
                response = "SAFE"

            cv2.putText(img, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 2)

            cv2.imshow("Fog Node - Fire Detection", img)
            cv2.waitKey(1)

            return response, 200

        return "Decode Failed", 400

    return "No Data", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)