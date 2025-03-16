import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from complaint_detection.utils.preprocessor import TextPreprocessor
from complaint_detection.src.train import train_model

def main():
    try:
        # Get the current directory and project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Set up paths
        data_path = os.path.join(project_root, 'complaint_detection', 'data', 'raw', 'conversations.csv')
        model_save_path = os.path.join(project_root, 'complaint_detection', 'models', 'saved_models')
        
        # Create model save directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize preprocessor with increased max_len
        preprocessor = TextPreprocessor(
            max_words=15000,  # Increased for larger vocabulary
            max_len=1000,     # Increased for longer sequences
            vocab_path=os.path.join(model_save_path, 'preprocessor.json')
        )
        
        logger.info(f"Data source: {data_path}")
        logger.info(f"Model save location: {model_save_path}")
        
        # Calculate class weights based on your data distribution
        n_complaint = 1500
        n_non_complaint = 5000
        total_samples = n_complaint + n_non_complaint
        
        # Class weights for balanced training
        class_weights = {
            0: 1.0,  # Weight for non-complaint class
            1: n_non_complaint / n_complaint  # Weight for complaint class
        }
        
        logger.info(f"Using class weights: {class_weights}")
        
        # Train model with updated parameters
        model, preprocessor, history = train_model(
            data_path=data_path,
            model_save_path=model_save_path,
            preprocessor=preprocessor,
            epochs=30,          # Increased epochs
            batch_size=32,      # Increased batch size for stability
            learning_rate=0.001,
            class_weights=class_weights
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {history}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 