from pathlib import Path
from ultralytics.models.sam import Predictor as SAMPredictor
import pandas as pd
from tqdm import tqdm
import yaml
import json
from shapely.geometry import Polygon
from shapely import to_geojson
from dvc.api import params_show

    
params = params_show(stages=['object_detection_training', 'footprint'])
imgsz = params["train"]["imgsz"]
segment_object_name = params["objects_name"]

# Initialize paths
base_path = Path(__file__).parents[2]
data_path = base_path / "data"
config_path = data_path / "xView" / "xView.yaml"
output_path = data_path / "output"
(output_path/ "masked_images").mkdir(parents=True, exist_ok=True)

for object_name in segment_object_name:
    (output_path / "masked_images" / object_name).mkdir(parents=True, exist_ok=True)
    
    
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
object_name_id_mapping = {v:k for k, v in config["names"].items()}

source_processed = list((data_path / "processed_data").glob("*.jpeg"))
detection_output_path = output_path / "objects.csv"
detection_output = pd.read_csv(detection_output_path)


# Create SAMPredictor
overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=imgsz, model="sam_b.pt",) 
predictor = SAMPredictor(overrides=overrides)

for file in tqdm(source_processed):
    predictor.set_image(file)  # set with image file
    detection = detection_output[(detection_output["image_name"] == file.name) & (detection_output["image_type"] == "processed")]
    # Segment each object of interest separately
    for object_name in segment_object_name:
        predictor.save_dir = output_path / "masked_images" / object_name
        detection_obj = detection[detection["class"] == object_name_id_mapping[object_name]]
        detection_boxes, detection_labels = detection_obj[["x1", "y1", "x2", "y2"]].values, detection_obj["class"].values
        results = predictor(bboxes=detection_boxes, labels=detection_labels, )  
        geojson = {'type': 'FeatureCollection', 'features': []}
        for i in range(len(results[0].masks)):
            feature = {'type': 'Feature', 
                        'properties': {},
                        'geometry': json.loads(to_geojson(Polygon(results[0].masks[i].xy[0])))}
            geojson['features'].append(feature) 
            
        with open(output_path / "masked_images" / object_name / f"{file.stem}.geojson", 'w') as f:
            json.dump(geojson, f, indent=2)
    predictor.reset_image()



        
