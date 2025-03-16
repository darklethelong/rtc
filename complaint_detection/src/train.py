import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from complaint_detection.utils.preprocessor import TextPreprocessor
from complaint_detection.models.complaint_classifier import ComplaintClassifier

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess text
        x = self.preprocessor.transform_text(text)
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def load_and_preprocess_data(data_path, preprocessor):
    """Load and preprocess the complaint data"""
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        # Fit preprocessor on training texts
        preprocessor.fit(df['text'].values)
        
        # Convert labels to numeric
        df['label'] = (df['label'] == 'complaint').astype(int)
        
        return df['text'].values, df['label'].values
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model, val_loader, device):
    """Evaluate model performance"""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in val_loader:
            metrics = model.validation_step(batch, device)
            all_metrics.append(metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics

def plot_evaluation_metrics(metrics_history, save_dir):
    """Plot training history"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    plt.close()
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.plot(metrics_history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        plt.title(f'Model {metric.upper()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_history.png'))
    plt.close()

def train_model(data_path, model_save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Training pipeline with comprehensive evaluation"""
    print(f"\nUsing device: {device}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        max_words=10000,
        max_len=200,
        vocab_path='domain_vocabulary.json'
    )
    
    print("\nLoading and preprocessing data...")
    # Load and preprocess data
    texts, labels = load_and_preprocess_data(data_path, preprocessor)
    
    print("\nDataset statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Split data
    test_size = min(0.2, 1/len(labels))  # Adjust test size for small datasets
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Create datasets
    train_dataset = ComplaintDataset(X_train, y_train, preprocessor)
    val_dataset = ComplaintDataset(X_val, y_val, preprocessor)
    
    # Create dataloaders
    batch_size = min(32, len(y_train))  # Adjust batch size for small datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Adjust based on your system
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Adjust based on your system
    )
    
    # Initialize model
    model = ComplaintClassifier(
        max_words=10000,
        max_len=200,
        embedding_dim=100,
        num_filters=128,
        lstm_units=64,
        num_heads=8,
        device=device
    )
    
    # Configure training
    optimizer, scheduler = model.configure_optimizers(learning_rate=0.001)
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            loss = model.training_step(batch, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({'train_loss': np.mean(train_losses)})
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Update history
        history['train_loss'].append(np.mean(train_losses))
        for key, value in val_metrics.items():
            history[key].append(value)
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1-Score: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(model_save_path, f"model_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_model(os.path.join(save_dir, "best_model.pt"))
            
            # Save preprocessor
            preprocessor.save_domain_knowledge(os.path.join(save_dir, "domain_vocabulary.json"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    # Plot and save metrics
    plot_evaluation_metrics(history, save_dir)
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    return model, preprocessor, history

def main():
    # Set paths
    data_path = "data/sample_data.csv"
    model_save_path = "models/"
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Train model with comprehensive evaluation
    model, preprocessor, history = train_model(data_path, model_save_path)
    
    # Print summary of results
    print("\nTraining History:")
    for metric, values in history.items():
        if metric.startswith('val_'):
            print(f"{metric[4:]}: {np.mean(values):.4f}")

if __name__ == "__main__":
    main() 