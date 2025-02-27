import os
import shutil
from pathlib import Path
from itertools import islice
import re

def sort_by_index(strs, pattern=r'(\d+).'):
    
    def get_index(_str):
        match = re.search(pattern, _str)
        return int(match.group(1)) if match else float('inf')
    
    return sorted(strs, key=get_index)

def get_images_from_dirs(dirs, exts={".jpg", ".png", ".jpeg"}):
    return sort_by_index([
        str(img) for d in map(Path, dirs) if d.exists() and d.is_dir()
        for img in d.iterdir() if img.is_file() and img.suffix.lower() in exts
    ])

def copy_images_to_temp(images, temp_dir):
    temp_dir.mkdir(exist_ok=True)
    list(map(lambda img: shutil.copy(img, temp_dir / img.name), images))

def zipify_image_dirs(*dirs, filename, n=100):
    """Zips up to `n` images from the given directories into a zip archive."""
    temp_dir = Path("temp_zip")
    
    print(*list(islice(get_images_from_dirs(dirs), n)), sep='\n')
    
    # (lambda imgs: (
    #     copy_images_to_temp(imgs, temp_dir),
    #     shutil.make_archive(filename, 'zip', temp_dir),
    #     shutil.rmtree(temp_dir)
    # ))(list(islice(get_images_from_dirs(dirs), n)))

    # print(f"Created {filename}.zip with {n} images.")


image_dirs = ['data/upscaled', 'data/watermark_upscaled', 
              'data/_alpha', 'data/_mask', 'data/_wm']

zipify_image_dirs(*image_dirs, filename="watermark_dataset", n=50)
