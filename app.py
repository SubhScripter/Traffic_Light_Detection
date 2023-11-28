import cv2
import time
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO  # Make sure to import YOLO from ultralytics

app = Flask(__name__)

# Load YOLOv8 model (make sure you have the YOLOv8 model files)
yolo_model = YOLO("best.pt")

# Video capture from webcam
camera = cv2.VideoCapture(0)  # 0 corresponds to the default webcam device

# Set the desired frames per second for 30 frames per second
desired_fps = 30
frame_delay = 1 / desired_fps

def generate_frames():
    while True:
        start_time = time.time()

        success, frame = camera.read()  # Read the frame from the webcam

        if not success:
            break
        else:
            # Perform object detection with YOLOv8
            objects = detect_objects(frame)
            
            # Perform voice output based on detected objects
            voice_output = generate_voice_output(objects)

            # Overlay the voice output on the frame
            cv2.putText(frame, f"Voice Output: {voice_output}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

def detect_objects(frame):
    # Preprocess the frame for YOLOv8
    yolo_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yolo_frame = cv2.resize(yolo_frame, (640, 480))  # Adjust the size as needed
    
    # Perform YOLOv8 object detection
    results = yolo_model.predict(yolo_frame)
    
    # Post-process the detections
    objects = []
    for box in results.xyxy[0]:
        x1, y1, x2, y2 = [round(coord.item()) for coord in box[:4]]
        class_id = int(box[5].item())
        prob = round(box[4].item(), 2)
        class_name = results.names[class_id]
        objects.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name, 'probability': prob})

    return objects

def generate_voice_output(objects):
    # Your logic to generate voice output based on detected objects
    # For example, check if a traffic light is detected and generate appropriate voice commands
    voice_output = "No relevant objects detected"
    for obj in objects:
        if obj['class'] == 'traffic_light':
            if obj['probability'] > 0.5:
                voice_output = "Traffic light detected. Please follow the signal."
                break
    return voice_output

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
