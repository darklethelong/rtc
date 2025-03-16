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
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc

from complaint_detection.utils.preprocessor import TextPreprocessor
from complaint_detection.models.complaint_classifier import ComplaintClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def load_and_preprocess_data(data_path, preprocessor):
    """Load and preprocess the data"""
    try:
        logger.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path)
        
        # Convert labels to binary
        df['label'] = (df['label'].str.lower() == 'complaint').astype(int)
        
        # Preprocess texts
        logger.info("Fitting preprocessor on training data...")
        preprocessor.fit(df['text'].values)
        
        logger.info("Transforming texts...")
        texts = preprocessor.transform_texts(df['text'].values)
        labels = df['label'].values
        
        return texts, labels
        
    except Exception as e:
        logger.error("Error loading data: %s", str(e))
        raise

def create_data_loaders(texts, labels, batch_size=32, test_size=0.2, val_size=0.2):
    """Create train, validation, and test data loaders"""
    # First split into train and temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=(test_size + val_size), random_state=42
    )
    
    # Then split temp into val and test
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    # Create datasets
    train_dataset = ComplaintDataset(train_texts, train_labels)
    val_dataset = ComplaintDataset(val_texts, val_labels)
    test_dataset = ComplaintDataset(test_texts, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_model(data_path, model_save_path, preprocessor=None, epochs=10, batch_size=32, learning_rate=0.001):
    """Train the complaint detection model"""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", device)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        texts, labels = load_and_preprocess_data(data_path, preprocessor)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            texts, labels, batch_size=batch_size
        )
        
        # Initialize model
        vocab_size = preprocessor.get_vocab_size()
        model = ComplaintClassifier(vocab_size=vocab_size).to(device)
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_steps = 0
            
            for batch_texts, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs.squeeze(), batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch_texts, batch_labels in val_loader:
                    batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                    outputs = model(batch_texts)
                    loss = criterion(outputs.squeeze(), batch_labels)
                    
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            
            logger.info(
                f'Epoch {epoch + 1}: '
                f'train_loss = {avg_train_loss:.4f}, '
                f'val_loss = {avg_val_loss:.4f}'
            )
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs(model_save_path, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss
                }, os.path.join(model_save_path, 'best_model.pt'))
                
                # Save preprocessor
                preprocessor.save_domain_knowledge(
                    os.path.join(model_save_path, 'preprocessor.json')
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Evaluate on test set
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                outputs = model(batch_texts)
                preds = torch.sigmoid(outputs.squeeze())
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        binary_preds = (all_preds >= 0.5).astype(int)
        report = classification_report(all_labels, binary_preds)
        logger.info("\nClassification Report:\n%s", report)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_save_path, 'roc_curve.png'))
        
        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_save_path, 'pr_curve.png'))
        
        return model, preprocessor, {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
    except Exception as e:
        logger.error("Error during training: %s", str(e))
        raise

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