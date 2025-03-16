import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        attended = torch.sum(attention_weights * x, dim=1)
        # attended shape: (batch_size, hidden_dim)
        return attended

class ComplaintClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,     # Increased for better representation
        hidden_dim=128,        # Increased for complexity
        num_filters=128,       # Increased for better feature extraction
        filter_sizes=[3, 4, 5, 6],  # Added larger filter for longer sequences
        dropout=0.4,           # Increased dropout for regularization
        class_weights=None     # Added class weights parameter
    ):
        super().__init__()
        
        # Store class weights
        self.class_weights = class_weights
        
        # Embedding layer with L2 regularization
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # CNN layers with batch normalization
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                    padding=fs // 2  # Added padding to handle sequence length
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for fs in filter_sizes
        ])
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=num_filters * len(filter_sizes),
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=2  # Added second LSTM layer for better sequence modeling
        )
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dim * 2)  # *2 for bidirectional
        
        # Additional fully connected layer
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer with L2 regularization
        self.fc2 = nn.Linear(hidden_dim, 1)  # Final output layer
        
        # L2 regularization weight
        self.l2_lambda = 0.01
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Embedding with dropout
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        # CNN with batch norm and dropout
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # Includes BatchNorm, ReLU, and Dropout
            conv_out = F.max_pool1d(
                conv_out,
                conv_out.shape[2]
            ).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(conv_out)
        
        # Concatenate CNN outputs
        conv_cat = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Reshape for LSTM
        conv_cat = conv_cat.unsqueeze(1)  # (batch_size, 1, num_filters * len(filter_sizes))
        conv_cat = conv_cat.repeat(1, x.shape[1], 1)  # (batch_size, seq_len, num_filters * len(filter_sizes))
        
        # BiLSTM
        lstm_out, _ = self.lstm(conv_cat)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Self-attention
        attended = self.attention(lstm_out)  # (batch_size, hidden_dim * 2)
        
        # Additional FC layer with batch norm
        fc1_out = self.fc1(attended)
        fc1_out = self.fc1_bn(fc1_out)
        fc1_out = F.relu(fc1_out)
        
        # Dropout
        dropped = self.dropout(fc1_out)
        
        # Output
        out = self.fc2(dropped)  # (batch_size, 1)
        
        # Add L2 regularization
        l2_reg = torch.tensor(0., requires_grad=True).to(out.device)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        
        return out, self.l2_lambda * l2_reg
        
    def configure_optimizers(self, learning_rate=0.001):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
        return optimizer, scheduler
    
    def training_step(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device).float()
        
        self.train()
        outputs, l2_reg = self(x)
        loss = F.binary_cross_entropy(outputs.squeeze(), y)
        
        return loss
    
    def validation_step(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device).float()
        
        self.eval()
        with torch.no_grad():
            outputs, _ = self(x)
            loss = F.binary_cross_entropy(outputs.squeeze(), y)
            preds = (outputs > 0.5).float()
            
            # Move tensors to CPU for metric calculation
            y_cpu = y.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            probs_cpu = outputs.cpu().numpy()
            
            metrics = {
                'val_loss': loss.item(),
                'accuracy': accuracy_score(y_cpu, preds_cpu),
                'precision': precision_score(y_cpu, preds_cpu, zero_division=0),
                'recall': recall_score(y_cpu, preds_cpu, zero_division=0),
                'f1': f1_score(y_cpu, preds_cpu, zero_division=0),
                'auc': roc_auc_score(y_cpu, probs_cpu) if len(np.unique(y_cpu)) > 1 else 0.5
            }
            
            return metrics
    
    def predict(self, x, device):
        x = torch.tensor(x, dtype=torch.long).to(device)
        
        self.eval()
        with torch.no_grad():
            outputs, _ = self(x)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
        return preds, probs
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.embedding.num_embeddings,
                'embedding_dim': self.embedding.embedding_dim,
                'hidden_dim': self.lstm.hidden_size,
                'num_filters': self.convs[0].out_channels,
                'filter_sizes': [conv.kernel_size[0] for conv in self.convs],
                'dropout': self.dropout.p
            }
        }, path)
    
    @classmethod
    def load_model(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model 