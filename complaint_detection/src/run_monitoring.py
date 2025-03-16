import os
import sys
from pathlib import Path
import glob
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from complaint_detection.utils.real_time_monitor import ComplaintMonitor
from complaint_detection.models.complaint_classifier import ComplaintClassifier
from complaint_detection.utils.preprocessor import TextPreprocessor

def get_latest_model(model_dir):
    """Get the path to the latest trained model"""
    result_dirs = glob.glob(os.path.join(model_dir, "results_*"))
    if not result_dirs:
        raise FileNotFoundError("No trained models found!")
    
    latest_dir = max(result_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_dir, "model")
    vocab_path = os.path.join(latest_dir, "domain_vocabulary.json")
    
    return model_path, vocab_path

def main():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    model_dir = os.path.join(project_root, 'complaint_detection', 'models', 'saved_models')
    
    try:
        # Get latest model and vocabulary
        model_path, vocab_path = get_latest_model(model_dir)
        print(f"Using model from: {model_path}")
        
        # Initialize components
        model = ComplaintClassifier.load_model(model_path)
        preprocessor = TextPreprocessor(vocab_path=vocab_path)
        monitor = ComplaintMonitor(window_size=5)
        
        print("\nReal-time Complaint Monitoring System")
        print("====================================")
        print("Enter conversation utterances (press Ctrl+C to exit)")
        print("Format: 'Agent: ' or 'Caller: ' followed by the message")
        
        current_conversation = []
        
        while True:
            try:
                utterance = input("\nEnter utterance: ").strip()
                
                if not utterance:
                    continue
                    
                if not (utterance.lower().startswith('agent:') or utterance.lower().startswith('caller:')):
                    print("Error: Utterance must start with 'Agent:' or 'Caller:'")
                    continue
                
                # Add timestamp and append to conversation
                timestamp = datetime.now().strftime("%H:%M:%S")
                timestamped_utterance = f"{timestamp} {utterance}"
                current_conversation.append(timestamped_utterance)
                
                # Process conversation
                conversation_text = '\n'.join(current_conversation[-4:])  # Use last 4 utterances
                processed_text, features = preprocessor.prepare_conversation(conversation_text)
                sequences = preprocessor.texts_to_sequences([processed_text])
                
                # Get prediction
                probability = model.predict(sequences)[0][0]
                
                # Update monitor
                monitor.update(probability, timestamp)
                
            except KeyboardInterrupt:
                print("\nExiting monitoring system...")
                break
            except Exception as e:
                print(f"Error processing utterance: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error initializing monitoring system: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 