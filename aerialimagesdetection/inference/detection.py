from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from PIL import Image
import wandb

# Initialize W&B run
output = (Path(__file__).parents[2] / "data" / "output")
output.mkdir(parents=True, exist_ok=True)

(output/ "annotated_images").mkdir(parents=True, exist_ok=True)

model_path = Path(__file__).parents[2] / "runs" / "detect" / "train2" / "weights" / "best.pt"
# Define path to directory containing images and videos for inference
source = Path(__file__).parents[2] / "data" / "upscaled_data"

objects_pd = pd.DataFrame(columns=["image_name", "class", "x1", "y1", "x2", "y2"])
# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)
# Add W&B callback for Ultralytics
# Run inference on the source - save results to the output directory 
results = model(source, stream=True)

# Show the results
for e, r in enumerate(results):
    boxes = r.boxes
    classes, boxes_coord = boxes.cls, boxes.xyxy
    image_name = Path(r.path).name
    objects_pd = objects_pd.append({"image_name": image_name, "class": classes, "x1": boxes_coord[0], "y1": boxes_coord[1], "x2": boxes_coord[2], "y2": boxes_coord[3]}, ignore_index=True)
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(str(output / 'annotated_images' / image_name))  # save image
    
objects_pd.to_csv(output / "objects.csv", index=False)

# Finish the W&B run
wandb.finish()