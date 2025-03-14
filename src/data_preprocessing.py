import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ConversationPreprocessor:
    def __init__(self, window_size=3):
        """
        Initialize the conversation preprocessor.
        
        Args:
            window_size (int): Number of utterances to consider as context window
        """
        self.window_size = window_size
        self.stop_words = set(stopwords.words('english'))
        self.max_sequence_length = 128
        self.tokenizer = None

    def clean_text(self, text):
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z0-9\s\']', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove speech-to-text artifacts like repeated words
        text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text)
        
        return text.strip()
    
    def split_conversation(self, text):
        """Split conversation into individual utterances."""
        # Split by speaker indicators (Agent: and Caller:)
        utterances = re.split(r'(Agent:|Caller:)', text)
        
        # Filter empty utterances and pair speaker with text
        processed_utterances = []
        current_speaker = None
        
        for utterance in utterances:
            utterance = utterance.strip()
            if utterance in ['Agent:', 'Caller:']:
                current_speaker = utterance[:-1]  # Remove colon
            elif utterance and current_speaker:
                processed_utterances.append({
                    'speaker': current_speaker,
                    'text': self.clean_text(utterance)
                })
        
        return processed_utterances
    
    def extract_features(self, utterances):
        """Extract features from a list of utterances."""
        features = []
        
        for utterance in utterances:
            # Count question marks
            question_count = utterance['text'].count('?')
            
            # Count words
            words = word_tokenize(utterance['text'])
            word_count = len(words)
            
            # Count unique words
            unique_word_count = len(set(words))
            
            # Ratio of caller utterances
            is_caller = 1 if utterance['speaker'] == 'Caller' else 0
            
            features.append({
                'text': utterance['text'],
                'speaker': utterance['speaker'],
                'question_count': question_count,
                'word_count': word_count,
                'unique_word_count': unique_word_count,
                'is_caller': is_caller
            })
        
        return features
    
    def create_windows(self, conversations, labels):
        """Create sliding windows of utterances with labels."""
        windows = []
        window_labels = []
        
        for i, (conversation, label) in enumerate(zip(conversations, labels)):
            utterances = self.split_conversation(conversation)
            if not utterances:
                continue
                
            features = self.extract_features(utterances)
            
            # Create sliding windows
            if len(features) >= self.window_size:
                for j in range(len(features) - self.window_size + 1):
                    window = features[j:j+self.window_size]
                    windows.append(window)
                    window_labels.append(label)
        
        return windows, window_labels
    
    def prepare_text_sequences(self, windows):
        """Convert text windows to sequence tensors."""
        # Extract text from each utterance in each window
        texts = []
        for window in windows:
            window_text = " ".join([utterance['text'] for utterance in window])
            texts.append(window_text)
        
        # Prepare tokenizer if not already fit
        if self.tokenizer is None:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length, 
            padding='post'
        )
        
        return padded_sequences
    
    def prepare_feature_arrays(self, windows):
        """Extract numerical features from windows."""
        features = []
        
        for window in windows:
            window_features = []
            
            # Features for the whole window
            caller_ratio = sum(1 for u in window if u['speaker'] == 'Caller') / len(window)
            question_ratio = sum(u['question_count'] for u in window) / len(window)
            
            # Add window-level features
            window_features.extend([caller_ratio, question_ratio])
            
            # Add utterance-level features
            for utterance in window:
                window_features.extend([
                    utterance['word_count'],
                    utterance['unique_word_count'],
                    utterance['word_count'] / (utterance['unique_word_count'] if utterance['unique_word_count'] > 0 else 1),
                    utterance['question_count'],
                    1 if utterance['speaker'] == 'Caller' else 0
                ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def load_and_process_data(self, file_path, test_size=0.2, val_size=0.1):
        """Load data, preprocess, and split into train, validation, and test sets."""
        # Load data
        df = pd.read_csv(file_path)
        
        # Create conversation windows
        windows, labels = self.create_windows(df['text'].values, df['label'].values)
        
        # Convert labels to binary
        binary_labels = np.array([1 if label == 'complaint' else 0 for label in labels])
        
        # Prepare text sequences and additional features
        text_sequences = self.prepare_text_sequences(windows)
        feature_arrays = self.prepare_feature_arrays(windows)
        
        # Split data into train+val and test
        X_train_val_text, X_test_text, X_train_val_feat, X_test_feat, y_train_val, y_test = train_test_split(
            text_sequences, 
            feature_arrays, 
            binary_labels, 
            test_size=test_size, 
            random_state=42,
            stratify=binary_labels
        )
        
        # Split train+val into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train_text, X_val_text, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
            X_train_val_text, 
            X_train_val_feat, 
            y_train_val, 
            test_size=val_size_adjusted, 
            random_state=42,
            stratify=y_train_val
        )
        
        return {
            'train': (X_train_text, X_train_feat, y_train),
            'val': (X_val_text, X_val_feat, y_val),
            'test': (X_test_text, X_test_feat, y_test),
            'tokenizer': self.tokenizer
        }
        
    def process_real_time_utterance(self, new_utterance, conversation_history):
        """Process a new utterance in real-time."""
        # Add the new utterance to conversation history
        conversation_history.append(new_utterance)
        
        # Keep only the most recent utterances based on window size
        if len(conversation_history) > self.window_size:
            conversation_history = conversation_history[-self.window_size:]
        
        # Extract features from the current window
        features = self.extract_features(conversation_history)
        
        # Create text input
        window_text = " ".join([utterance['text'] for utterance in features])
        sequence = self.tokenizer.texts_to_sequences([window_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=self.max_sequence_length, 
            padding='post'
        )
        
        # Create feature input
        feature_array = self.prepare_feature_arrays([features])
        
        return padded_sequence, feature_array, conversation_history 