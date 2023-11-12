from pathlib import Path
from ultralytics import SAM

output = (Path(__file__).parents[2] / "data" / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "footprints.csv").touch()

(output/ "masked_images").mkdir(parents=True, exist_ok=True)



# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

source = Path(__file__).parents[2] / "data" / "upscaled_data"

# # Run inference with bboxes prompt
# model(str(source / "image_1.jpeg"), bboxes=[954.1512,  243.0841, 1090.0707,  320.6259], save_dir=output / "masked_images")
# breakpoint()


from ultralytics.data.annotator import auto_annotate
det_model = Path(__file__).parents[2] / "runs" / "detect" / "train2" / "weights" / "best.pt"
auto_annotate(data=source, det_model=det_model, sam_model='sam_b.pt', output_dir=output / "masked_images")

breakpoint()
