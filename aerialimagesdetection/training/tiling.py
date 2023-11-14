import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics.data.utils import autosplit
from tqdm import tqdm

def adjust_and_save_bboxes(label_file, left, upper, right, lower, orig_width, orig_height, output_folder, tile_name, x, y):
    with open(label_file, 'r') as file:
        bboxes = file.readlines()

    new_bboxes = []

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = map(float, bbox.split())
        
        class_id = int(class_id)

        # Convert normalized coordinates to absolute coordinates based on original image size
        abs_x_center, abs_y_center = x_center * orig_width, y_center * orig_height
        abs_width, abs_height = width * orig_width, height * orig_height
        
        x1, y1 = abs_x_center - abs_width / 2, abs_y_center - abs_height / 2
        x2, y2 = abs_x_center + abs_width / 2, abs_y_center + abs_height / 2

        # Check if the bounding box is within the tile
        if left <= x1 and upper <= y1 and right >= x2 and lower >= y2:
            # Adjust coordinates to the tile

            # Normalize the coordinates for the tile
            new_x_center, new_y_center = abs_x_center / (right - left), abs_y_center / (lower - upper)
            new_width, new_height = abs_width / (right - left), abs_height / (lower - upper)
            
            new_x_center = new_x_center - x
            new_y_center = new_y_center - y

            # Add the adjusted bounding box to the list
            new_bboxes.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")

    # Write the adjusted bounding boxes to a new file
    with open(output_folder / "labels" / f"{tile_name}.txt", 'w') as file:
        for bbox in new_bboxes:
            file.write(bbox)


def process_images(image_folder, label_folder, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / 'images').mkdir(parents=True, exist_ok=True)
    (output_folder / 'labels').mkdir(parents=True, exist_ok=True)
    for image_file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        label_file = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')
        
        if not os.path.exists(label_file):
            continue

        img = Image.open(image_path)
        width, height = img.size
        num_tiles_x = width // 640
        num_tiles_y = height // 640

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Define the tile boundaries
                left = i * 640
                upper = j * 640
                right = (i + 1) * 640
                lower = (j + 1) * 640
                tile = img.crop((left, upper, right, lower))

                # Save the tile
                tile_name = f"id_{os.path.splitext(image_file)[0]}_part_{i}_{j}"
                tile_path = output_folder / "images" / f"{tile_name}.jpg"
                tile.save(tile_path)

                # Adjust and save bounding boxes for this tile
                adjust_and_save_bboxes(label_file, left, upper, right, lower, width, height, output_folder, tile_name, i, j)

# Call this function with the appropriate paths
def draw_bboxes(image_path, label_path):
    # Load the image
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img)

    # Load the label file and draw bounding boxes
    with open(label_path, 'r') as file:
        bboxes = file.readlines()

    for bbox in bboxes:
        class_id, x, y, w, h = map(float, bbox.split())

        # Convert normalized coordinates to absolute coordinates
        width, height = img.size
        abs_x = (x * width) - (w * width / 2)
        abs_y = y * height - (h * height / 2)
        abs_w = w * width
        abs_h = h * height
        
        # Create a Rectangle patch
        rect = patches.Rectangle((abs_x, abs_y), abs_w, abs_h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig('test.png')

if __name__ == "__main__":
    process_images('data/xView/images', 'data/xView/labels', 'data/xView/tiled')
    autosplit('data/xView/tiled/images')
