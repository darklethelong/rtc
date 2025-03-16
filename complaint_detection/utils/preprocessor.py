import re
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=200, vocab_path=None):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
            
        self.stop_words = set(stopwords.words('english'))
        
        # Load domain-specific vocabulary and patterns
        self._load_domain_knowledge(vocab_path)
        
    def _load_domain_knowledge(self, vocab_path=None):
        """Load or initialize domain-specific knowledge"""
        self.domain_knowledge = {
            'banking_terms': {
                'account', 'transaction', 'deposit', 'withdrawal', 'balance',
                'credit', 'debit', 'transfer', 'payment', 'fee', 'charge',
                'overdraft', 'interest', 'loan', 'mortgage', 'statement',
                'pin', 'atm', 'branch', 'online', 'mobile', 'banking',
                'savings', 'checking', 'wire', 'ach', 'direct deposit'
            },
            'complaint_indicators': {
                'issue', 'problem', 'error', 'mistake', 'wrong', 'incorrect',
                'failed', 'failure', 'denied', 'reject', 'delay', 'late',
                'missing', 'lost', 'stolen', 'fraud', 'unauthorized',
                'dissatisfied', 'unhappy', 'angry', 'frustrated', 'upset'
            },
            'masked_patterns': {
                r'\b[x]+\b',  # Basic x pattern
                r'\b\d{4}[-\s]?(?:\d{4}[-\s]?){2}\d{4}\b',  # Card numbers
                r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Emails
            }
        }

        # Add domain-specific stop words
        self.domain_stop_words = {
            'please', 'thank', 'thanks', 'hello', 'hi', 'hey', 'okay', 'ok', 
            'yes', 'no', 'bye', 'goodbye', 'sir', 'madam', 'mr', 'mrs', 'ms'
        }
        self.stop_words.update(self.domain_stop_words)

        # Load custom vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                custom_vocab = json.load(f)
                for category, terms in custom_vocab.items():
                    if category in self.domain_knowledge:
                        self.domain_knowledge[category].update(terms)

    def save_domain_knowledge(self, path):
        """Save domain knowledge for future use"""
        with open(path, 'w') as f:
            json.dump(self.domain_knowledge, f, indent=2)

    def update_domain_knowledge(self, new_terms, category):
        """Update domain knowledge with new terms"""
        if category in self.domain_knowledge:
            self.domain_knowledge[category].update(new_terms)

    def clean_text(self, text):
        """Clean and normalize text with enhanced masking"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle masked information
        for pattern in self.domain_knowledge['masked_patterns']:
            text = re.sub(pattern, 'MASKED_INFO', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
        
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_features(self, text):
        """Extract rich features from text"""
        # Basic features
        basic_features = {
            'has_exclamation': '!' in text,
            'has_question': '?' in text,
            'word_count': len(text.split()),
            'masked_info_count': text.count('MASKED_INFO'),
        }
        
        # Domain-specific features
        words = set(text.lower().split())
        domain_features = {
            'banking_term_count': len(words & self.domain_knowledge['banking_terms']),
            'complaint_indicator_count': len(words & self.domain_knowledge['complaint_indicators']),
            'has_banking_terms': bool(words & self.domain_knowledge['banking_terms']),
            'has_complaint_indicators': bool(words & self.domain_knowledge['complaint_indicators'])
        }
        
        return {**basic_features, **domain_features}
    
    def preprocess_text(self, text):
        """Enhanced preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        # Add special tokens for important domain terms
        tokens = [
            f'BANKING_{token.upper()}' if token in self.domain_knowledge['banking_terms']
            else f'COMPLAINT_{token.upper()}' if token in self.domain_knowledge['complaint_indicators']
            else token
            for token in tokens
        ]
        
        return ' '.join(tokens)
    
    def prepare_conversation(self, conversation):
        """Process conversation with enhanced context awareness"""
        lines = conversation.split('\n')
        caller_utterances = []
        agent_utterances = []
        context = []
        
        for line in lines:
            if line.strip():
                if line.lower().startswith('caller:'):
                    caller_text = line.split(':', 1)[1].strip()
                    caller_utterances.append(caller_text)
                    context.append(('caller', caller_text))
                elif line.lower().startswith('agent:'):
                    agent_text = line.split(':', 1)[1].strip()
                    agent_utterances.append(agent_text)
                    context.append(('agent', agent_text))
        
        # Combine caller utterances with context
        caller_text = ' [SEP] '.join(caller_utterances)
        
        # Process the combined text
        processed_text = self.preprocess_text(caller_text)
        features = self.extract_features(caller_text)
        
        # Add contextual features
        features['total_turns'] = len(context)
        features['caller_turns'] = len(caller_utterances)
        features['agent_turns'] = len(agent_utterances)
        
        return processed_text, features
    
    def prepare_features(self, features_list):
        """Convert features to numpy array with enhanced features"""
        feature_array = np.array([
            [
                int(f['has_exclamation']),
                int(f['has_question']),
                f['word_count'],
                f['masked_info_count'],
                f['banking_term_count'],
                f['complaint_indicator_count'],
                int(f['has_banking_terms']),
                int(f['has_complaint_indicators']),
                f['total_turns'],
                f['caller_turns'],
                f['agent_turns']
            ]
            for f in features_list
        ])
        return feature_array

    def fit_tokenizer(self, texts):
        """Fit tokenizer with domain knowledge"""
        # Add domain terms to vocabulary
        domain_terms = list(self.domain_knowledge['banking_terms'] | 
                          self.domain_knowledge['complaint_indicators'])
        self.tokenizer.fit_on_texts(texts + domain_terms)
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences with handling of unknown tokens"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post') 