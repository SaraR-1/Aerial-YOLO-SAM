from ultralytics import YOLO
from pathlib import Path
from PIL import Image

output = (Path(__file__).parents[2] / "data" / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "objects.csv").touch()

(output/ "annotated_images").mkdir(parents=True, exist_ok=True)


model_path = Path(__file__).parents[2] / "runs" / "detect" / "train2" / "weights" / "best.pt"
# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)

# Define path to directory containing images and videos for inference
source = Path(__file__).parents[2] / "data" / "upscaled_data"
# Run inference on the source - save results to the output directory 
results = model(source, stream=True)

# Show the results
for e, r in enumerate(results):
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(str(output / 'annotated_images' / f'results_{e}.jpg'))  # save image