from ultralytics import YOLO
import os

# Correct the path to your downloaded YOLOv8 model (for example, yolov8n.pt)
model_path = "D:/Data Science/VS CODE/Open ai/yoloenv/weights/yolov8n.pt"  # Update with the correct model filename

# Check if the model file exists
if os.path.exists(model_path):
    print(f"Model found at {model_path}")
    # Load the model
    model = YOLO(model_path)  # Loading the YOLO model with the correct path
else:
    print(f"Model not found at {model_path}. Please verify the path.")
