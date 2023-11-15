import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.utils.ops import xyxy2xywhn

CLASS_MAP = {
    0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1, 
    10: -1, 11: 0, 12: 0, 13: 0, 14: -1, 15: -1, 16: -1, 17: 1, 18: 1, 19: 1, 
    20: 1, 21: 1, 22: -1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 
    30: -1, 31: -1, 32: 1, 33: 3, 34: 1, 35: 1, 36: 1, 37: 1, 38: 3, 39: -1, 
    40: 2, 41: 2, 42: 2, 43: -1, 44: 2, 45: 2, 46: -1, 47: 2, 48: -1, 49: 2, 
    50: 2, 51: 2, 52: 2, 53: 1, 54: -1, 55: -1, 56: 1, 57: -1, 58: -1, 59: -1, 
    60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: -1, 66: 1, 67: -1, 68: -1, 69: -1, 
    70: -1, 71: -1, 72: 4, 73: 4, 74: 4, 75: -1, 76: 4, 77: 4, 78: -1, 79: -1, 
    80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: 4, 87: -1, 88: -1, 89: -1,
    90: -1, 91: -1, 92: -1, 93: -1, 94: -1
}   

def convert_labels(fname):
    # Convert xView geoJSON labels to YOLO format
    path = fname.parent
    with open(fname) as f:
        print(f'Loading {fname}...')
        data = json.load(f)

    # Make dirs
    labels = Path(path / 'labels')
    os.system(f'rm -rf {labels}')
    labels.mkdir(parents=True, exist_ok=True)

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting {fname}'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id'].split('.')[0]
            file = path / 'images' / f'{id}.jpg'
            if file.exists():  # 1395.tif missing
                try:
                    box = np.array([int(num) for num in p['bounds_imcoords'].split(",")])
                    assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
                    class_label = p['type_id']
                    class_label = CLASS_MAP[int(class_label)]  # xView class to 0-60
                    assert 59 >= class_label >= 0, f'incorrect class index {class_label}'

                    # Write YOLO label
                    if id not in shapes:
                        shapes[id] = Image.open(file).size
                    box = xyxy2xywhn(box[None].astype(float), w=shapes[id][0], h=shapes[id][1], clip=True)
                    with open((labels / id).with_suffix('.txt'), 'a') as f:
                        f.write(f"{class_label} {' '.join(f'{x:.6f}' for x in box[0])}\n")  # write label.txt
                except AssertionError as e:
                    print(f'Assertion Broke for {file}')
                except Exception as e:
                    breakpoint()

if __name__ == "__main__":
    dataset_root_dir = Path(__file__).parents[2] / "data" / "xView"
    convert_labels(dataset_root_dir / 'xView_train.geojson')