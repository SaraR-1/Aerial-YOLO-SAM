from PIL import Image
from pathlib import Path
from tqdm import tqdm
        
def convert_to_jpeg(input_folder, output_folder, quality=85):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(input_folder.glob('*.tif')), desc='Converting to JPEG'):
        if (output_folder / f'{img_path.stem}.jpg').exists():
            continue
        with Image.open(img_path) as img:
            # Convert to RGB if necessary (JPEG doesn't support alpha channel)
            if img.mode in ["RGBA", "P"]:
                img = img.convert("RGB")

            output_path = output_folder / f'{img_path.stem}.jpg'
            img.save(output_path, 'JPEG', quality=quality)

# Replace 'path/to/directory' with the path to your directory of .tif files
convert_to_jpeg('data/xView/images', 'data/xView/images/jpg')
