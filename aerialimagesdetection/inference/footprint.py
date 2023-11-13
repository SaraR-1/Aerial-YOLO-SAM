from pathlib import Path
from ultralytics import SAM
from ultralytics.data.annotator import auto_annotate

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np
import random

def read_polygons_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    polygons = {}
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        vertices = data[1:]
        if class_id not in polygons:
            polygons[class_id] = []
        polygons[class_id].append(vertices)
    return polygons

def get_color_map(num_classes):
    # Generate distinct colors for each class in BGR format
    return {class_id: [random.randint(0, 255) for _ in range(3)] for class_id in range(num_classes)}

def plot_polygons_on_image(image_path, polygons, filename=None, opacity=0.5,):
    # Read the original image
    image = cv2.imread(str(image_path))
    height, width, _ = image.shape

    color_map = get_color_map(50)
    
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Process each class and its polygons
    for class_id, class_polygons in polygons.items():
        if class_id != 48:
            continue
        for polygon in class_polygons:
            points = []
            num_vertices = int(len(polygon) / 2)
            for i in range(num_vertices):
                x, y = float(polygon[2 * i]), float(polygon[2 * i + 1])
                x_img, y_img = int(x * width), int(y * height)
                points.append([x_img, y_img])
            points = np.array([points], np.int32)
            cv2.fillPoly(overlay, [points], color=(50, 100, 255))
    
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    # Save or display the image
    if filename:
        cv2.imwrite(str(filename), image)
    


output = (Path(__file__).parents[2] / "data" / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "footprints.csv").touch()

(output/ "masked_images").mkdir(parents=True, exist_ok=True)

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

source = Path(__file__).parents[2] / "data" / "upscaled_data"

det_model = Path(__file__).parents[2] / "runs" / "detect" / "train4" / "weights" / "best.pt"

for file in source.iterdir():
    if not (output / "masked_images" / f"{file.name}.txt").exists():
        auto_annotate(data=file, det_model=det_model, sam_model='sam_b.pt', output_dir=output / "masked_images")
    annotated = output / "masked_images" / f"{file.stem}.txt"
    polygons = read_polygons_from_file(annotated)
    plot_polygons_on_image(file, polygons, output / "masked_images" / f"{file.stem}.png")
