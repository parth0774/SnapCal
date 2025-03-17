import os
import shutil
from collections import defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(filepath, src, dest):
    """
    Prepare training or testing data by copying images to appropriate class folders.
    
    Args:
        filepath (str): Path to the text file containing image paths
        src (str): Source directory containing all images
        dest (str): Destination directory to organize images by class
    """
    # Convert paths to Path objects for better cross-platform compatibility
    src_path = Path(src)
    dest_path = Path(dest)
    
    classes_images = defaultdict(list)
    
    # Read file paths and organize by class
    try:
        with open(filepath, 'r') as txt:
            paths = [read.strip() for read in txt.readlines()]
            for p in paths:
                food_class, image_name = p.split('/')
                classes_images[food_class].append(f"{image_name}.jpg")
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return
        
    # Create directories and copy images
    for food_class in classes_images.keys():
        logger.info(f"Processing class: {food_class}")
        
        # Create destination directory if it doesn't exist
        class_dir = dest_path / food_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images with progress tracking
        total_images = len(classes_images[food_class])
        for i, img in enumerate(classes_images[food_class], 1):
            if i % 100 == 0:
                logger.info(f"Progress for {food_class}: {i}/{total_images}")
            
            src_file = src_path / food_class / img
            dest_file = class_dir / img
            
            try:
                shutil.copy(src_file, dest_file)
            except FileNotFoundError:
                logger.warning(f"Could not find or copy: {src_file}")
    
    logger.info(f"Copying completed for {filepath}")

def main():
    """Execute data preparation for both training and testing sets"""
    base_dir = Path('food-101')
    
    logger.info("Starting data preparation...")
    
    # Prepare training data
    prepare_data(base_dir / 'meta' / 'train.txt', 
                 base_dir / 'images', 
                 base_dir / 'train')
    
    # Prepare testing data
    prepare_data(base_dir / 'meta' / 'test.txt', 
                 base_dir / 'images', 
                 base_dir / 'test')
    
    logger.info("Data preparation completed!")

if __name__ == "__main__":
    main()
