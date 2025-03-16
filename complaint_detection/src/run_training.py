import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from complaint_detection.src.train import train_model

def main():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'complaint_detection', 'data', 'raw', 'conversations.csv')
    model_save_path = os.path.join(project_root, 'complaint_detection', 'models', 'saved_models')
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    print("Starting model training...")
    print(f"Using data from: {data_path}")
    print(f"Models will be saved to: {model_save_path}")
    
    try:
        # Train the model
        train_model(data_path, model_save_path)
        print("\nTraining completed successfully!")
        print(f"Check {model_save_path} for the trained model and evaluation results.")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 