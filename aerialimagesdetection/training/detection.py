from ultralytics import YOLO
from pathlib import Path
from dvc.api import params_show

params = params_show(stages='object_detection_training')["train"]

# Load a model
model = YOLO('yolov8m.pt', )  # load a pretrained model (recommended for training)
parent_folder = Path(__file__).parents[2] 
config_path = parent_folder / "data" / "xView" / "xView.yaml"
output = parent_folder / "YOLOv8"
(output / "detection_model").mkdir(parents=True, exist_ok=True)

# Set training configuration
config = {
    "data": str(config_path),  # Path to data YAML file
    "epochs": params["epochs"],  
    "batch": params["batch"], 
    "project": "YOLOv8",  # Directory to save trained models
    "name": "detection_model",  # Experiment name
    "exist_ok": True,  
    "val": params["val"],  # Validate model after each epoch
    "imgsz": params["imgsz"],  
}

# Train the model
model.train(**config)