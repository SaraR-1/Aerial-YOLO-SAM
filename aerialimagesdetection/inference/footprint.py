from pathlib import Path

output = (Path(__file__).parents[1] / "output")
output.mkdir(parents=True, exist_ok=True)

(output / "footprints.csv").touch()

(output/ "masked_images").mkdir(parents=True, exist_ok=True)