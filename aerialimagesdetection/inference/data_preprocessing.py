import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

    
def main():
  input = (Path(__file__).parents[2] / "data" / "raw_data")
  output = (Path(__file__).parents[2] / "data" / "processed_data")
  output.mkdir(parents=True, exist_ok=True)

  images = [f for f in Path(input).glob( "*.jpeg")]
  for img_name in tqdm(images):
    img = cv2.imread(str(img_name))

    min_val = np.min(img)
    max_val = np.max(img)
    img_stretched = (img - min_val) * 255.0 / (max_val - min_val) ## Scale the pixel values to the full dynamic range of the image
    img_stretched = np.uint8(img_stretched)

    cv2.imwrite(str(output / img_name.name), img_stretched)
    
if __name__ == "__main__":
  main()