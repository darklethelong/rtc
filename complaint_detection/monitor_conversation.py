import sys
import time
from datetime import datetime
from utils.preprocessor import TextPreprocessor
from utils.real_time_monitor import ComplaintMonitor
from models.complaint_classifier import ComplaintClassifier

class ConversationMonitor:
    def __init__(self, model_path, window_size=10):
        """Initialize the conversation monitor"""
        self.preprocessor = TextPreprocessor()
        self.model = ComplaintClassifier.load_model(model_path)
        self.monitor = ComplaintMonitor(window_size=window_size)
        
    def process_utterance(self, text, timestamp=None):
        """Process a single utterance and update the monitor"""
        # Preprocess the text
        processed_text, features = self.preprocessor.prepare_conversation(f"Caller: {text}")
        
        # Convert to sequence
        sequence = self.preprocessor.texts_to_sequences([processed_text])
        
        # Get prediction
        probability = float(self.model.predict(sequence)[0][0])
        
        # Update the monitor
        self.monitor.update(probability, timestamp)
        
        return probability
        
    def process_conversation(self, conversation_file, real_time=True, delay=2):
        """Process a full conversation file"""
        with open(conversation_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.lower().startswith('caller:'):
                # Extract timestamp if present
                timestamp = None
                if '[' in line and ']' in line:
                    timestamp = line[line.find('[')+1:line.find(']')]
                    text = line[line.find(']')+1:].split(':', 1)[1].strip()
                else:
                    text = line.split(':', 1)[1].strip()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Process the utterance
                probability = self.process_utterance(text, timestamp)
                
                # Print current status
                print(f"\nTimestamp: {timestamp}")
                print(f"Utterance: {text}")
                print(f"Complaint Probability: {probability:.2f}")
                print(f"Complaint Level: {self.monitor.get_complaint_level(probability)}")
                print("-" * 50)
                
                # Add delay for real-time simulation
                if real_time:
                    time.sleep(delay)
                    
    def save_session(self, filepath):
        """Save the current monitoring session"""
        self.monitor.save_session(filepath)
        
    def load_session(self, filepath):
        """Load a previous monitoring session"""
        self.monitor.load_session(filepath)

def main():
    if len(sys.argv) < 3:
        print("Usage: python monitor_conversation.py <model_path> <conversation_file> [--no-realtime]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    conversation_file = sys.argv[2]
    real_time = '--no-realtime' not in sys.argv
    
    # Initialize monitor
    monitor = ConversationMonitor(model_path)
    
    try:
        # Process conversation
        monitor.process_conversation(conversation_file, real_time=real_time)
        
        # Save session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor.save_session(f"monitoring_session_{timestamp}.json")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {str(e)}")
        
if __name__ == "__main__":
    main() 