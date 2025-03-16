import os
import sys
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from complaint_detection.src.train import train_model

if __name__ == "__main__":
    # Set paths
    data_path = os.path.join(project_root, 'complaint_detection', 'data', 'raw', 'conversations.csv')
    model_save_path = os.path.join(project_root, 'complaint_detection', 'models', 'saved_models')
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Train model
        model, preprocessor, history = train_model(
            data_path=data_path,
            model_save_path=model_save_path,
            device=device
        )
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise 