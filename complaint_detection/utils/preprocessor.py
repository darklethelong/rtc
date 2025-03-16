import re
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter
import torch

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, max_words=15000, max_len=1000, vocab_path=None):
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
        else:
            sequence = sequence[:self.max_len]
        
        return torch.tensor(sequence, dtype=torch.long)
    
    def transform_texts(self, texts):
        """Convert multiple texts to sequences"""
        sequences = [self.transform_text(text) for text in texts]
        return torch.stack(sequences)
    
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
                # Account-related terms
                'account', 'balance', 'statement', 'transaction', 'history',
                'savings', 'checking', 'deposit', 'withdrawal', 'transfer',
                'joint account', 'business account', 'personal account',
                
                # Payment-related terms
                'payment', 'transfer', 'wire', 'ach', 'direct deposit',
                'bill pay', 'autopay', 'recurring payment', 'standing order',
                'payee', 'beneficiary', 'remittance', 'swift', 'routing number',
                
                # Card-related terms
                'credit card', 'debit card', 'card number', 'pin', 'cvv',
                'expiration date', 'chip', 'contactless', 'swipe', 'declined',
                'authorization', 'limit', 'credit limit', 'available credit',
                
                # Fees and charges
                'fee', 'charge', 'surcharge', 'commission', 'interest',
                'rate', 'apr', 'annual fee', 'maintenance fee', 'overdraft fee',
                'minimum balance', 'insufficient funds', 'nsf', 'late fee',
                
                # Loan-related terms
                'loan', 'mortgage', 'refinance', 'interest rate', 'principal',
                'collateral', 'down payment', 'installment', 'credit score',
                'application', 'approval', 'term', 'amortization',
                
                # Digital banking
                'online banking', 'mobile banking', 'app', 'website', 'login',
                'password', 'security code', 'two-factor', 'authentication',
                'notification', 'alert', 'e-statement', 'paperless',
                
                # Service channels
                'atm', 'branch', 'teller', 'representative', 'customer service',
                'call center', 'phone banking', 'automated system', 'ivr',
                'queue', 'wait time', 'callback', 'appointment',
                
                # Security-related
                'fraud', 'suspicious', 'unauthorized', 'security', 'freeze',
                'lock', 'unlock', 'dispute', 'claim', 'investigation',
                'verification', 'identity', 'password reset'
            },
            'complaint_indicators': {
                # Direct complaint words
                'issue', 'problem', 'error', 'mistake', 'wrong', 'incorrect',
                'failed', 'failure', 'denied', 'reject', 'delay', 'late',
                'missing', 'lost', 'stolen', 'fraud', 'unauthorized',
                
                # Emotional indicators
                'dissatisfied', 'unhappy', 'angry', 'frustrated', 'upset',
                'disappointed', 'annoyed', 'concerned', 'worried', 'anxious',
                'stressed', 'confused', 'inconvenienced', 'unacceptable',
                
                # Urgency indicators
                'urgent', 'immediate', 'asap', 'emergency', 'critical',
                'important', 'pressing', 'deadline', 'overdue', 'escalate',
                'supervisor', 'manager', 'complaint department',
                
                # Service quality indicators
                'poor service', 'bad experience', 'unprofessional', 'rude',
                'incompetent', 'unhelpful', 'unresponsive', 'slow', 'waiting',
                'long time', 'never received', 'still waiting', 'again',
                
                # Resolution demands
                'resolve', 'fix', 'correct', 'rectify', 'address', 'solve',
                'refund', 'reimburse', 'compensate', 'credit back', 'reverse',
                'waive', 'remove', 'adjust', 'investigate',
                
                # Escalation indicators
                'complaint', 'grievance', 'dispute', 'formal complaint',
                'legal', 'lawyer', 'attorney', 'regulatory', 'report',
                'better business bureau', 'bbb', 'cfpb', 'authorities'
            },
            'service_context': {
                # Time-related context
                'yesterday', 'last week', 'previous month', 'recent',
                'repeatedly', 'multiple times', 'again', 'still', 'pending',
                
                # Amount-related context
                'dollars', 'cents', 'amount', 'total', 'balance', 'money',
                'funds', 'available', 'pending', 'hold', 'cleared',
                
                # Process-related context
                'process', 'procedure', 'policy', 'requirement', 'verification',
                'confirmation', 'approval', 'rejection', 'status', 'update'
            }
        }

        # Add domain-specific stop words while preserving important context
        self.domain_stop_words = {
            # Basic conversational words
            'please', 'thank', 'thanks', 'hello', 'hi', 'hey', 'bye', 
            'goodbye', 'good morning', 'good afternoon', 'good evening',
            
            # Titles and honorifics
            'sir', 'madam', 'mr', 'mrs', 'ms', 'dr', 'prof',
            
            # Basic responses (preserve context-specific ones)
            'yes', 'no', 'maybe', 'okay', 'ok', 'alright', 'sure',
            
            # Common verbs (preserve action-specific ones)
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            
            # Articles and basic prepositions
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'down', 'over', 'under'
        }
        self.stop_words.update(self.domain_stop_words)

        # Load custom vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                custom_vocab = json.load(f)
                for category, terms in custom_vocab.items():
                    if category in self.domain_knowledge:
                        self.domain_knowledge[category].update(terms)

    def extract_features(self, text):
        """Extract rich features from text"""
        # Basic features
        basic_features = {
            'has_exclamation': '!' in text,
            'has_question': '?' in text,
            'word_count': len(text.split()),
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
        processed_lines = [self.preprocess_text(line) for line in lines]
        return ' '.join(processed_lines) 