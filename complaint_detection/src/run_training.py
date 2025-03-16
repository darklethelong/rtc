import os
import sys
import logging
from complaint_detection.utils.preprocessor import TextPreprocessor
from complaint_detection.src.train import train_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(
            max_words=10000,
            max_len=200,
            vocab_path=os.path.join(model_save_path, 'preprocessor.json')
        )
        
        logger.info(f"Data source: {data_path}")
        logger.info(f"Model save location: {model_save_path}")
        
        # Train model
        model, preprocessor, history = train_model(
            data_path=data_path,
            model_save_path=model_save_path,
            preprocessor=preprocessor,
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {history}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 