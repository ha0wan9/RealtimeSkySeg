from pathlib import Path
import shutil
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np

def convert_annotation_masks(src_dir: str, mask_suffix: str):
    """
    Recursively converts all annotation mask files in each subdirectory under the
    specified source directory to binary format, where 0 represents non-sky regions
    and 255 represents sky regions.
    """
    # Convert src_dir to a Path object
    src_path = Path(src_dir)

    # Traverse the directory tree under src_dir
    for mask_path in tqdm(list(src_path.glob('**/*' + mask_suffix)), desc=f'Converting annotation masks in {src_dir}'):
        # Open the mask image
        with Image.open(mask_path) as mask_img:
            # Convert the mask image to a NumPy array
            mask = np.array(mask_img)

            # Set non-sky pixels to 0 and sky pixels to 1
            mask[mask != 10] = 0
            mask[mask == 10] = 1

            # Save the mask image
            mask_img = Image.fromarray(mask)
            mask_img.save(mask_path)

def move_data_files(src_dir: str, img_suffix: str):
    """
    Recursively moves all image files in each subdirectory under the specified
    source directory to the source directory, and removes the original subdirectories.
    """
    # Convert src_dir to a Path object
    src_path = Path(src_dir)

    # Traverse the directory tree under src_dir
    for dir_path in tqdm(list(src_path.glob('**/*')), desc=f'Moving files in {src_dir}'):
        # Skip files and the root directory
        if not dir_path.is_dir() or dir_path == src_path:
            continue

        # Create the target directory
        target_dir = src_path

        # Loop through all image files in the directory
        for img_path in dir_path.glob('*' + img_suffix):
            # Construct the target file path
            target_path = target_dir.joinpath(img_path.name)

            # Move the file to the target directory
            shutil.move(img_path, target_path)

        # Remove the blank directory
        dir_path.rmdir()

def main():
    # Source image directory paths
    img_dirs = ['data/datasets/cityscapes/leftImg8bit/train', 
                'data/datasets/cityscapes/leftImg8bit/val', 
                'data/datasets/cityscapes/leftImg8bit/test']

    # Image file suffix
    img_suffix = '_leftImg8bit.png'

    # Source annotation directory paths
    ann_dirs = ['data/datasets/cityscapes/gtFine/train', 
                'data/datasets/cityscapes/gtFine/val', 
                'data/datasets/cityscapes/gtFine/test']

    # Annotation file suffix
    ann_suffix = '_gtFine_labelTrainIds.png'

    # Set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

    # Move image files
    for img_dir in img_dirs:
        logging.info(f'Moving image files in {img_dir}')
        move_data_files(img_dir, img_suffix)

    # Move annotation files
    for ann_dir in ann_dirs:
        logging.info(f'Moving annotation files in {ann_dir}')
        move_data_files(ann_dir, ann_suffix)

    # Convert annotation masks
    for ann_dir in ann_dirs:
        logging.info(f'Converting annotation masks in {ann_dir}')
        convert_annotation_masks(ann_dir, ann_suffix)

if __name__ == '__main__':
    main()