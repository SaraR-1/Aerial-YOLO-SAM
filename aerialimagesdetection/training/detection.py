from ultralytics import YOLO
from pathlib import Path

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

config_path = Path(__file__).parents[2] / "data" / "xView.yaml"

# Train the model
results = model.train(data=str(config_path), epochs=10, imgsz=640, )