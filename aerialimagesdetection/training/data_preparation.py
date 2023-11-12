import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.data.utils import autosplit
from ultralytics.utils.ops import xyxy2xywhn

dataset_root_dir = Path(__file__).parents[2] / "data" / "xView"


def convert_labels(fname=Path('xView/xView_train.geojson')):
    # Convert xView geoJSON labels to YOLO format
    path = fname.parent
    with open(fname) as f:
        print(f'Loading {fname}...')
        data = json.load(f)

    # Make dirs
    labels = Path(path / 'labels' / 'train')
    os.system(f'rm -rf {labels}')
    labels.mkdir(parents=True, exist_ok=True)

    # xView classes 11-94 to 0-59
    xview_class2index = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
                        12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
                        29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46,
                        47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59]

    shapes = {}
    for feature in tqdm(data['features'], desc=f'Converting {fname}'):
        p = feature['properties']
        if p['bounds_imcoords']:
            id = p['image_id']
            file = path / 'train_images' / id
            if file.exists():  # 1395.tif missing
                try:
                    box = np.array([int(num) for num in p['bounds_imcoords'].split(",")])
                    assert box.shape[0] == 4, f'incorrect box shape {box.shape[0]}'
                    class_label = p['type_id']
                    class_label = xview_class2index[int(class_label)]  # xView class to 0-60
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

convert_labels(dataset_root_dir / 'xView_train.geojson')

# Move images
images = Path(dataset_root_dir / 'images')
images.mkdir(parents=True, exist_ok=True)
Path(dataset_root_dir / 'train_images').rename(dataset_root_dir / 'images' / 'train')

# Split
autosplit(dataset_root_dir / 'images' / 'train')
