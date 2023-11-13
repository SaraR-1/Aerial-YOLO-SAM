import cv2
from pathlib import Path

    
input = (Path(__file__).parents[2] / "data" / "raw_data")
output = (Path(__file__).parents[2] / "data" / "upscaled_data")
output.mkdir(parents=True, exist_ok=True)

model_path = (Path(__file__).parents[2] / "EDSR_x3.pb")

images = [f for f in Path(input).glob( "*.jpeg")]
 
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(model_path))
sr.setModel("edsr", 3) # set the model by passing the value and the upsampling ratio

for img_name in images:
    # Read image
    image = cv2.imread(str(img_name))
    result = sr.upsample(image) # upscale the input image
    # Save the image
    cv2.imwrite(str(output / img_name.name), result)