from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def process_images(model, source_images, output_dir, objects_pd, prefix=""):
    results = model(source_images, stream=True)
    for e, result in tqdm(enumerate(results), total=len(source_images)):
        boxes = result.boxes
        classes, boxes_coord = boxes.cls.cpu(), boxes.xyxy.cpu()
        image_name = Path(result.path).name

        new_rows = [{
            "image_name": image_name, 
            "image_type": prefix,
            "class": int(classes[i].item()), 
            "x1": boxes_coord[i][0].item(), 
            "y1": boxes_coord[i][1].item(), 
            "x2": boxes_coord[i][2].item(), 
            "y2": boxes_coord[i][3].item()
        } for i in range(len(classes))]

        objects_pd = pd.concat([objects_pd, pd.DataFrame(new_rows)], ignore_index=True)
        
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(output_dir / f"{prefix}_{image_name}")
        
    return objects_pd

# Initialize paths
base_path = Path(__file__).parents[2]
data_path = base_path / "data"
output_path = data_path / "output"
output_path.mkdir(parents=True, exist_ok=True)
annotated_images_path = output_path / "annotated_images"
annotated_images_path.mkdir(parents=True, exist_ok=True)

model_path = base_path / "YOLOv8" / "detection_model" / "weights" / "best.pt"

# Load model
model = YOLO(model_path)

# Data sources
source_processed = list((data_path / "processed_data").glob("*.jpeg"))
source_raw = list((data_path / "raw_data").glob("*.jpeg"))

# DataFrame for storing results
objects_pd = pd.DataFrame(columns=["image_name", "image_type", "class", "x1", "y1", "x2", "y2"])

# Process images
objects_pd = process_images(model, source_processed, annotated_images_path, objects_pd, "processed")
objects_pd = process_images(model, source_raw, annotated_images_path, objects_pd, "raw")

# Save results
objects_pd.to_csv(output_path / "objects.csv", index=False)
