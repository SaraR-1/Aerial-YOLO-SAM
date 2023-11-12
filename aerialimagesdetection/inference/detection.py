from pathlib import Path

output = (Path(__file__).parents[2] / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "objects.csv").touch()

(output/ "annotated_images").mkdir(parents=True, exist_ok=True)