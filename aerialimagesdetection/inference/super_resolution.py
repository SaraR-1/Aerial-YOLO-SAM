import cv2
from pathlib import Path
from tqdm import tqdm

    
input = (Path(__file__).parents[2] / "data" / "raw_data")
output = (Path(__file__).parents[2] / "data" / "upscaled_data")
output.mkdir(parents=True, exist_ok=True)

model_path = (Path(__file__).parents[2] / "EDSR_x3.pb")
 
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(model_path))
sr.setModel("edsr", 3) # set the model by passing the value and the upsampling ratio

images = [f for f in Path(input).glob( "*.jpeg")]
for img_name in tqdm(images):
    # Read image
    image = cv2.imread(str(img_name))
    result = sr.upsample(image) # upscale the input image
    # Produce 3x3 non-overlapping crops of the image of the original size within the upscaled image. 3=upsampling ratio
    # Original image dimensions
    orig_height, orig_width = image.shape[:2]

    # Iterate and crop
    for i in range(3):  # Vertical crops
        for j in range(3):  # Horizontal crops
            x_start = j * orig_width
            y_start = i * orig_height
            cropped_image = result[y_start:y_start + orig_height, x_start:x_start + orig_width]
            crop_name = f"cropped_{i}_{j}_{img_name.name}"
            cv2.imwrite(str(output / crop_name), cropped_image)
