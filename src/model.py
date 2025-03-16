import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, LSTM, 
    Dense, Dropout, Bidirectional, BatchNormalization,
    Concatenate, GlobalMaxPooling1D
)
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# Configure TensorFlow to be more resilient with GPU issues
try:
    # Attempt to configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    else:
        print("No GPUs found, using CPU")
except Exception as e:
    print(f"GPU configuration failed: {e}. Using CPU instead.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ComplaintDetector:
    def __init__(
        self, 
        vocab_size=10000,
        embedding_dim=100,
        max_sequence_length=128, 
        num_features=17,
        window_size=3
    ):
        """
        Initialize the complaint detector model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            max_sequence_length (int): Maximum length of input sequences
            num_features (int): Number of numerical features
            window_size (int): Number of utterances in each window
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_features = num_features
        self.window_size = window_size
        self.model = None
        
    def build_model(self):
        """Build and compile the hybrid CNN-LSTM model."""
        # Text sequence input branch
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            name='embedding'
        )(text_input)
        
        # Multiple parallel convolutional layers with different filter sizes
        conv_blocks = []
        for filter_size in [2, 3, 4]:
            conv = Conv1D(
                filters=64,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                name=f'conv_{filter_size}'
            )(embedding)
            pool = GlobalMaxPooling1D(name=f'pool_{filter_size}')(conv)
            conv_blocks.append(pool)
        
        # Concatenate CNN outputs
        conv_features = Concatenate(name='concatenate_conv')(conv_blocks)
        conv_dropout = Dropout(0.2, name='conv_dropout')(conv_features)
        
        # Bidirectional LSTM layer on the embedding
        lstm = Bidirectional(
            LSTM(64, return_sequences=True, name='lstm'),
            name='bidirectional'
        )(embedding)
        lstm_pool = GlobalMaxPooling1D(name='lstm_pool')(lstm)
        lstm_dropout = Dropout(0.2, name='lstm_dropout')(lstm_pool)
        
        # Combine CNN and LSTM features
        text_features = Concatenate(name='text_features')([conv_dropout, lstm_dropout])
        text_batch_norm = BatchNormalization(name='text_batch_norm')(text_features)
        
        # Numerical features input branch
        numerical_input = Input(shape=(self.num_features,), name='numerical_input')
        numerical_batch_norm = BatchNormalization(name='numerical_batch_norm')(numerical_input)
        numerical_dense = Dense(32, activation='relu', name='numerical_dense')(numerical_batch_norm)
        numerical_dropout = Dropout(0.2, name='numerical_dropout')(numerical_dense)
        
        # Combine text and numerical features
        combined = Concatenate(name='combined_features')([text_batch_norm, numerical_dropout])
        
        # Final dense layers
        dense1 = Dense(64, activation='relu', name='dense1')(combined)
        dropout1 = Dropout(0.3, name='dropout1')(dense1)
        dense2 = Dense(32, activation='relu', name='dense2')(dropout1)
        dropout2 = Dropout(0.3, name='dropout2')(dense2)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dropout2)
        
        # Create model
        model = Model(
            inputs=[text_input, numerical_input],
            outputs=output,
            name='complaint_detector'
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(
        self, 
        X_train_text, 
        X_train_features, 
        y_train, 
        X_val_text=None, 
        X_val_features=None, 
        y_val=None, 
        epochs=10, 
        batch_size=32,
        class_weights=None
    ):
        """
        Train the model on the given data.
        
        Args:
            X_train_text: Text sequences for training
            X_train_features: Numerical features for training
            y_train: Labels for training
            X_val_text: Text sequences for validation
            X_val_features: Numerical features for validation
            y_val: Labels for validation
            epochs: Number of epochs to train
            batch_size: Batch size for training
            class_weights: Optional weights for imbalanced classes
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Prepare validation data if provided
        validation_data = None
        if X_val_text is not None and X_val_features is not None and y_val is not None:
            validation_data = (
                [X_val_text, X_val_features],
                y_val
            )
        
        # Train model
        history = self.model.fit(
            [X_train_text, X_train_features],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=2,
                    min_lr=0.00001
                )
            ]
        )
        
        return history
    
    def evaluate(self, X_test_text, X_test_features, y_test):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        # Get predictions
        y_pred_probs = self.model.predict([X_test_text, X_test_features])
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Compute metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get test loss and metrics
        test_results = self.model.evaluate([X_test_text, X_test_features], y_test, verbose=0)
        metrics = {name: value for name, value in zip(self.model.metrics_names, test_results)}
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'metrics': metrics,
            'predictions': y_pred_probs
        }
    
    def predict(self, X_text, X_features):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        return self.model.predict([X_text, X_features])
    
    def save_model(self, model_dir):
        """Save the model to the given directory."""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, 'complaint_detector.h5'))
        
    def load_model(self, model_path):
        """Load the model from the given path."""
        self.model = tf.keras.models.load_model(model_path)
        return self.model 