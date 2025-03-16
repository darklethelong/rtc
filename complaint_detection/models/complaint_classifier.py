import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.layer_norm(x + attn_output)

class ComplaintClassifier(nn.Module):
    def __init__(
        self, 
        max_words=10000,
        max_len=200,
        embedding_dim=100,
        num_filters=128,
        lstm_units=64,
        num_heads=8,
        dropout_rate=0.5,
        use_pretrained_embeddings=False,
        embedding_matrix=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Embedding layer
        if use_pretrained_embeddings and embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix),
                freeze=True,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                max_words,
                embedding_dim,
                padding_idx=0
            )
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.MaxPool1d(2)
            ) for kernel_size in [3, 4, 5]
        ])
        
        # RNN layers
        self.bilstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            bidirectional=True,
            batch_first=True
        )
        
        self.bigru = nn.GRU(
            lstm_units * 2,
            lstm_units // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # Self-attention
        self.attention = SelfAttention(lstm_units, num_heads)
        
        # Dense layers
        cnn_out_dim = num_filters * 3  # 3 kernel sizes
        rnn_out_dim = lstm_units
        total_features = cnn_out_dim + rnn_out_dim
        
        self.fc1 = nn.Linear(total_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        
        self.fc3 = nn.Linear(128, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        self.output = nn.Linear(128, 1)
        
        self.to(device)
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, emb_dim]
        
        # CNN branch
        x_conv = x.permute(0, 2, 1)  # [batch, emb_dim, seq_len]
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_conv)
            conv_out = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(conv_out)
        
        # RNN branch
        lstm_out, _ = self.bilstm(x)
        gru_out, _ = self.bigru(lstm_out)
        attention_out = self.attention(gru_out)
        rnn_out = F.adaptive_avg_pool1d(attention_out.permute(0, 2, 1), 1).squeeze(-1)
        
        # Concatenate features
        concat = torch.cat(conv_outputs + [rnn_out], dim=1)
        
        # Dense layers with residual connection
        x = self.fc1(concat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        residual = x
        x = self.fc3(x)
        x = self.layer_norm(x + residual)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x
        
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
        outputs = self(x)
        loss = F.binary_cross_entropy(outputs.squeeze(), y)
        
        return loss
    
    def validation_step(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device).float()
        
        self.eval()
        with torch.no_grad():
            outputs = self(x)
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
            outputs = self(x)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
        return preds, probs
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'max_words': self.max_words,
                'max_len': self.max_len,
                'embedding_dim': self.embedding_dim,
                'num_filters': self.num_filters,
                'lstm_units': self.lstm_units,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate
            }
        }, path)
    
    @classmethod
    def load_model(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model 