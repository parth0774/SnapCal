import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras.src import load_model
from keras.src import image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_class_list(path):
    """
    Create a sorted list of food classes from directory structure.
    
    Args:
        path (str): Path to images directory
    
    Returns:
        list: Sorted list of class names
    """
    classes = []
    path = Path(path)
    
    # Get only the directories (which are class names)
    for item in path.iterdir():
        if item.is_dir():
            classes.append(item.name)
    
    return sorted(classes)

def preprocess_image(img_path, target_size=(299, 299)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        img_path (str): Path to image file
        target_size (tuple): Target size for the image
    
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        return img_array, img
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {str(e)}")
        return None, None

def predict_class(model_path, image_paths, class_list, show=True, top_n=3):
    """
    Predict food classes for given images.
    
    Args:
        model_path (str): Path to the trained model
        image_paths (list): List of image paths
        class_list (list): List of class names
        show (bool): Whether to display the image with prediction
        top_n (int): Number of top predictions to show
    
    Returns:
        dict: Dictionary mapping image paths to predictions
    """
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return {}
    
    results = {}
    
    for img_path in image_paths:
        logger.info(f"Processing image: {img_path}")
        
        img_array, original_img = preprocess_image(img_path)
        if img_array is None:
            continue
            
        # Get predictions
        preds = model.predict(img_array)
        
        # Get top N predictions
        top_indices = preds[0].argsort()[-top_n:][::-1]
        top_preds = [(class_list[i], float(preds[0][i])) for i in top_indices]
        
        # Store results
        results[img_path] = top_preds
        
        if show:
            # Display image with predictions
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
            
            # Show prediction probabilities
            plt.subplot(1, 2, 2)
            y_pos = np.arange(len(top_preds))
            bars = plt.barh(y_pos, [pred[1] * 100 for pred in top_preds])
            plt.yticks(y_pos, [pred[0] for pred in top_preds])
            plt.xlabel('Probability (%)')
            plt.title('Top Predictions')
            
            # Add percentage text
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                         f'{top_preds[i][1]*100:.1f}%', 
                         va='center')
            
            plt.tight_layout()
            plt.savefig(f"prediction_{os.path.basename(img_path)}.png")
            plt.show()
    
    return results

def main():
    """Main function to demonstrate prediction functionality"""
    # Path to your trained model
    model_path = 'models/final_model.h5'
    
    # Get list of all model files
    model_dir = Path('models')
    if model_dir.exists():
        model_files = list(model_dir.glob('*.h5'))
        if model_files:
            # Get the most recent model file
            model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
            logger.info(f"Using most recent model: {model_path}")
    
    # Get food classes
    class_list = create_class_list('food-101/images')
    logger.info(f"Found {len(class_list)} food classes")
    
    # Test images
    test_images = [
        'test_images/caesar.jpg',
        'test_images/beet2.jpg',
        'test_images/pancake.jpg'
    ]
    
    # Make predictions
    logger.info("Making predictions...")
    results = predict_class(model_path, test_images, class_list, show=True)
    
    # Print results
    logger.info("\nPrediction Results:")
    for img_path, preds in results.items():
        logger.info(f"\nImage: {os.path.basename(img_path)}")
        for i, (class_name, prob) in enumerate(preds, 1):
            logger.info(f"  #{i}: {class_name} - {prob*100:.2f}%")

if __name__ == "__main__":
    main()