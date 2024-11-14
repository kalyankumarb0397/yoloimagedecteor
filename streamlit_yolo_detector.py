# Install required packages if not already installed
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install OpenCV and ultralytics
try:
    import cv2
except ImportError:
    install_package("opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    install_package("ultralytics")

import numpy as np
import random
import streamlit as st

# opening the file in read mode with specified encoding
with open(r'/workspaces/yoloimagedecteor/coco.txt', "r", encoding='utf-8') as my_file:
    data = my_file.read()

# split the text by newline
class_list = data.split("\n")

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load YOLO model
model = YOLO("/workspaces/yoloimagedecteor/yolo11n.pt", "11n")  # Replace with actual YOLO model path

# Video capture parameters
frame_wid = 640
frame_hyt = 480

# Streamlit setup
st.title("Real-Time Object Detection with YOLO")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    # Create a temporary file path for the uploaded video
    video_path = f"temp_video.{video_file.name.split('.')[-1]}"
    
    # Save uploaded video to the temporary path
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video")
        exit()

    stframe = st.empty()

    while True:
        ret, frame = cap.read()

        if not ret:
            st.warning("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize the frame for optimization
        frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Predict on the frame
        detect_params = model.predict(source=[frame], conf=0.45, save=True)

        DP = detect_params[0].numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

        # Convert frame to RGB before displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Optional: stop if video ends
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            st.success("Video processed successfully")
            break

    cap.release()
else:
    st.info("Upload a video to start object detection")
