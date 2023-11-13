from ultralytics import YOLO
from pathlib import Path
import wandb

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

config_path = Path(__file__).parents[2] / "data" / "xView.yaml"

# Train the model
results = model.train(data=str(config_path), epochs=300, imgsz=640, batch=4, val=False)