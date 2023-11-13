from pathlib import Path
from ultralytics import SAM
from ultralytics.utils.plotting import Annotator

output = (Path(__file__).parents[2] / "data" / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "footprints.csv").touch()

(output/ "masked_images").mkdir(parents=True, exist_ok=True)



# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

source = Path(__file__).parents[2] / "data" / "upscaled_data"

from ultralytics.data.annotator import auto_annotate
det_model = Path(__file__).parents[2] / "runs" / "detect" / "train2" / "weights" / "best.pt"
auto_annotate(data=source, det_model=det_model, sam_model='sam_b.pt', output_dir=output / "masked_images")

Annotator(im = str(source / "image_1.jpeg")

breakpoint()
