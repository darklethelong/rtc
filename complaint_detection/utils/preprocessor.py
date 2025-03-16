import re
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=200, vocab_path=None):
        self.max_words = max_words
        self.max_len = max_len
        self.vocab_path = vocab_path
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load domain vocabulary if exists
        if vocab_path and os.path.exists(vocab_path):
            self.load_domain_knowledge(vocab_path)
    
    def get_wordnet_pos(self, word, tag):
        """Map POS tag to first character lemmatize() accepts"""
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag[0], wordnet.NOUN)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Lemmatize with correct POS tag
        lemmatized = [
            self.lemmatizer.lemmatize(
                word,
                self.get_wordnet_pos(word, tag)
            ) for word, tag in pos_tags if word not in self.stop_words
        ]
        
        return lemmatized
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Reset word counts
        self.word_counts = Counter()
        
        # Process all texts
        for text in texts:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            
            # Update word counts
            self.word_counts.update(tokens)
        
        # Build vocabulary using most common words
        vocab_words = ['<PAD>', '<UNK>'] + [
            word for word, _ in self.word_counts.most_common(self.max_words - 2)
        ]
        
        # Create word to index mapping
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def transform_text(self, text):
        """Convert text to sequence of indices"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        # Convert to indices
        sequence = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens[:self.max_len]
        ]
        
        # Pad sequence
        if len(sequence) < self.max_len:
            sequence += [self.word2idx['<PAD>']] * (self.max_len - len(sequence))
        
        return sequence
    
    def transform_texts(self, texts):
        """Convert multiple texts to sequences"""
        return [self.transform_text(text) for text in texts]
    
    def get_vocab_size(self):
        """Get size of vocabulary"""
        return len(self.word2idx)
    
    def save_domain_knowledge(self, path):
        """Save vocabulary and word counts"""
        domain_knowledge = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': dict(self.word_counts)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(domain_knowledge, f, indent=2)
    
    def load_domain_knowledge(self, path):
        """Load vocabulary and word counts"""
        with open(path, 'r') as f:
            domain_knowledge = json.load(f)
        
        self.word2idx = domain_knowledge['word2idx']
        self.idx2word = {int(k): v for k, v in domain_knowledge['idx2word'].items()}
        self.word_counts = Counter(domain_knowledge['word_counts'])

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