import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import tensorflow as tf
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from complaint_detection.utils.preprocessor import TextPreprocessor
from complaint_detection.models.complaint_classifier import ComplaintClassifier

def evaluate_model(model, X_val, y_val, threshold=0.5):
    """Comprehensive model evaluation"""
    # Get predictions
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    metrics = classification_report(y_val, y_pred, output_dict=True)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Combine all metrics
    evaluation = {
        'classification_report': metrics,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'curves': {
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr': {'precision': precision.tolist(), 'recall': recall.tolist()}
        }
    }
    
    return evaluation

def plot_evaluation_metrics(evaluation, save_dir):
    """Plot and save evaluation metrics"""
    # Create evaluation plots directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        evaluation['curves']['roc']['fpr'],
        evaluation['curves']['roc']['tpr'],
        label=f'ROC curve (AUC = {evaluation["roc_auc"]:.2f})'
    )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        evaluation['curves']['pr']['recall'],
        evaluation['curves']['pr']['precision'],
        label=f'PR curve (AUC = {evaluation["pr_auc"]:.2f})'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()
    
    # Plot confusion matrix
    metrics = evaluation['classification_report']
    cm = np.array([
        [metrics['0']['support'] - metrics['0']['support'] * metrics['0']['recall'],
         metrics['0']['support'] * metrics['0']['recall']],
        [metrics['1']['support'] - metrics['1']['support'] * metrics['1']['recall'],
         metrics['1']['support'] * metrics['1']['recall']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def cross_validate(X_sequences, X_features, y, n_splits=5):
    """Perform cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_sequences, y), total=n_splits)):
        # Split data
        X_train_seq, X_val_seq = X_sequences[train_idx], X_sequences[val_idx]
        X_train_feat, X_val_feat = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize and train model
        model = ComplaintClassifier(
            max_words=10000,
            max_len=200,
            embedding_dim=100,
            num_filters=128,
            lstm_units=64,
            num_heads=8
        )
        model.compile_model(learning_rate=0.001)
        
        history = model.train(
            X_train_seq,
            y_train,
            validation_data=(X_val_seq, y_val),
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        evaluation = evaluate_model(model, X_val_seq, y_val)
        cv_results.append({
            'fold': fold + 1,
            'evaluation': evaluation,
            'history': {
                metric: values for metric, values in history.history.items()
            }
        })
    
    return cv_results

def load_and_preprocess_data(data_path, preprocessor):
    """Load and preprocess data with enhanced error handling"""
    try:
        # Read data
        df = pd.read_csv(data_path)
        
        # Process texts and extract features
        processed_texts = []
        features_list = []
        
        for text in tqdm(df['text'], desc="Processing texts"):
            processed_text, features = preprocessor.prepare_conversation(text)
            processed_texts.append(processed_text)
            features_list.append(features)
        
        # Convert labels
        labels = (df['label'] == 'complaint').astype(int)
        
        # Fit and transform texts
        preprocessor.fit_tokenizer(processed_texts)
        X_sequences = preprocessor.texts_to_sequences(processed_texts)
        
        # Prepare additional features
        X_features = preprocessor.prepare_features(features_list)
        
        return X_sequences, X_features, labels
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        raise

def train_model(data_path, model_save_path):
    """Enhanced training pipeline with comprehensive evaluation"""
    # Initialize preprocessor with domain knowledge
    preprocessor = TextPreprocessor(
        max_words=10000,
        max_len=200,
        vocab_path='domain_vocabulary.json'
    )
    
    # Load and preprocess data
    X_sequences, X_features, y = load_and_preprocess_data(data_path, preprocessor)
    
    # Perform cross-validation
    cv_results = cross_validate(X_sequences, X_features, y)
    
    # Train final model on full training set
    X_train_seq, X_val_seq, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        X_sequences, X_features, y, test_size=0.2, random_state=42
    )
    
    model = ComplaintClassifier(
        max_words=10000,
        max_len=200,
        embedding_dim=100,
        num_filters=128,
        lstm_units=64,
        num_heads=8
    )
    model.compile_model(learning_rate=0.001)
    
    # Train model
    history = model.train(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=10,
        batch_size=32
    )
    
    # Evaluate final model
    evaluation = evaluate_model(model, X_val_seq, y_val)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(model_save_path, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(results_dir, "model")
    model.save_model(model_path)
    
    # Save preprocessor vocabulary
    preprocessor.save_domain_knowledge(os.path.join(results_dir, "domain_vocabulary.json"))
    
    # Save evaluation metrics
    plot_evaluation_metrics(evaluation, results_dir)
    
    # Save all results
    results = {
        'cross_validation': cv_results,
        'final_evaluation': evaluation,
        'training_history': history.history
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    return model, preprocessor, results

def main():
    # Set paths
    data_path = "data/sample_data.csv"
    model_save_path = "models/"
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Train model with comprehensive evaluation
    model, preprocessor, results = train_model(data_path, model_save_path)
    
    # Print summary of results
    print("\nCross-validation Results:")
    cv_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    
    for cv_result in results['cross_validation']:
        metrics = cv_result['evaluation']['classification_report']['weighted avg']
        for metric in cv_metrics:
            cv_metrics[metric].append(metrics[metric])
    
    for metric, values in cv_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean:.3f} (±{std:.3f})")

if __name__ == "__main__":
    main() 