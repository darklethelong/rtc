import os
import argparse
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import sys
import json

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import ConversationPreprocessor
from src.model import ComplaintDetector

def plot_training_history(history, save_path=None):
    """Plot training history and optionally save it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def create_confusion_matrix_plot(conf_matrix, classes, save_path=None):
    """Create and save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def main(args):
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ConversationPreprocessor(window_size=args.window_size)
    
    # Load and preprocess data
    print(f"Loading data from {args.data_path}...")
    data = preprocessor.load_and_process_data(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    X_train_text, X_train_feat, y_train = data['train']
    X_val_text, X_val_feat, y_val = data['val']
    X_test_text, X_test_feat, y_test = data['test']
    tokenizer = data['tokenizer']
    
    # Calculate class weights if data is imbalanced
    class_weights = None
    if args.use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = {i: w for i, w in zip(classes, weights)}
        print(f"Class weights: {class_weights}")
    
    # Print data shapes
    print(f"Train shapes: Text {X_train_text.shape}, Features {X_train_feat.shape}, Labels {y_train.shape}")
    print(f"Validation shapes: Text {X_val_text.shape}, Features {X_val_feat.shape}, Labels {y_val.shape}")
    print(f"Test shapes: Text {X_test_text.shape}, Features {X_test_feat.shape}, Labels {y_test.shape}")
    
    # Build model
    print("Building and training model...")
    vocab_size = len(tokenizer.word_index) + 1  # +1 for OOV token
    num_features = X_train_feat.shape[1]
    
    detector = ComplaintDetector(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        max_sequence_length=args.max_sequence_length,
        num_features=num_features,
        window_size=args.window_size
    )
    
    model = detector.build_model()
    print(model.summary())
    
    # Train model
    history = detector.train(
        X_train_text, X_train_feat, y_train,
        X_val_text, X_val_feat, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = detector.evaluate(X_test_text, X_test_feat, y_test)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in evaluation['metrics'].items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nClassification Report:")
    for class_name, metrics in evaluation['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"Class {class_name}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        # Convert numpy arrays to lists
        eval_results = {
            'classification_report': evaluation['classification_report'],
            'confusion_matrix': evaluation['confusion_matrix'].tolist(),
            'metrics': evaluation['metrics']
        }
        json.dump(eval_results, f, indent=4)
    
    # Plot and save confusion matrix
    create_confusion_matrix_plot(
        evaluation['confusion_matrix'],
        classes=['Non-Complaint', 'Complaint'],
        save_path=os.path.join(args.output_dir, 'plots', 'confusion_matrix.png')
    )
    
    # Plot and save training history
    plot_training_history(
        history,
        save_path=os.path.join(args.output_dir, 'plots', 'training_history.png')
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, 'models', 'complaint_detector.h5')
    detector.save_model(os.path.dirname(model_path))
    print(f"Model saved to {model_path}")
    
    # Save tokenizer
    import pickle
    tokenizer_path = os.path.join(args.output_dir, 'models', 'tokenizer.pickle')
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a complaint detection model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory for models and plots')
    parser.add_argument('--window_size', type=int, default=3, help='Number of utterances in each window')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of the embedding vectors')
    parser.add_argument('--max_sequence_length', type=int, default=128, help='Maximum length of input sequences')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data to use for validation')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    args = parser.parse_args()
    
    main(args) 