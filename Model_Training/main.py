import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the food classification project"""
    parser = argparse.ArgumentParser(description='Food Classification Project')
    parser.add_argument('--mode', type=str, choices=['prep', 'train', 'predict', 'all'], 
                        default='all', help='Mode to run (prep, train, predict, all)')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of training epochs')
    parser.add_argument('--fine-tune', action='store_true', 
                        help='Enable fine-tuning of the base model')
    parser.add_argument('--model-path', type=str, default=None, 
                        help='Path to model for prediction')
    parser.add_argument('--images', nargs='+', default=None, 
                        help='Images to predict')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('test_images').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Execute based on mode
    if args.mode in ['prep', 'all']:
        logger.info("Running data preparation...")
        from Data_Preparation import main as prep_main
        prep_main()
    
    if args.mode in ['train', 'all']:
        logger.info("Running model training...")
        from Model_Builder import train_model
        train_model(epochs=args.epochs, batch_size=args.batch_size, fine_tune=args.fine_tune)
    
    if args.mode in ['predict', 'all']:
        logger.info("Running prediction...")
        from Accuracy_Test import main as predict_main
        predict_main()
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()