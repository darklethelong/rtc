import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, LSTM, 
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, concatenate, Bidirectional,
    GRU, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

class ComplaintClassifier:
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
        embedding_matrix=None
    ):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_matrix = embedding_matrix
        self.model = self._build_model()
        
    def _build_cnn_branch(self, embedding_output, kernel_sizes=[3, 4, 5]):
        conv_outputs = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(
                self.num_filters,
                kernel_size,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform'
            )(embedding_output)
            conv = BatchNormalization()(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv_outputs.append(conv)
        return conv_outputs
    
    def _build_rnn_branch(self, embedding_output):
        # Bidirectional LSTM layer
        lstm = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True)
        )(embedding_output)
        
        # GRU layer for capturing different temporal patterns
        gru = Bidirectional(
            GRU(self.lstm_units // 2, return_sequences=True)
        )(lstm)
        
        # Multi-Head Self Attention
        attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.lstm_units // self.num_heads
        )(gru, gru)
        
        # Add & Norm
        attention = LayerNormalization()(attention + gru)
        
        return attention

    def _build_model(self):
        # Input layer
        inputs = Input(shape=(self.max_len,))
        
        # Embedding layer
        if self.use_pretrained_embeddings and self.embedding_matrix is not None:
            embedding = Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[self.embedding_matrix],
                trainable=False
            )(inputs)
        else:
            embedding = Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_len
            )(inputs)
        
        # CNN Branch
        conv_outputs = self._build_cnn_branch(embedding)
        
        # RNN Branch with Self-Attention
        rnn_output = self._build_rnn_branch(embedding)
        
        # Global Pooling for all branches
        pooled_outputs = []
        for conv_output in conv_outputs:
            pooled = GlobalAveragePooling1D()(conv_output)
            pooled_outputs.append(pooled)
        
        pooled_rnn = GlobalAveragePooling1D()(rnn_output)
        pooled_outputs.append(pooled_rnn)
        
        # Concatenate all features
        concat = concatenate(pooled_outputs)
        
        # Dense layers with residual connections
        dense1 = Dense(256, activation='relu')(concat)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(self.dropout_rate * 0.8)(dense2)
        
        # Residual connection
        dense3 = Dense(128, activation='relu')(dense2)
        dense3 = LayerNormalization()(dense3 + dense2)
        dense3 = Dropout(self.dropout_rate * 0.5)(dense3)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(dense3)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1')
            ]
        )
        
    def get_callbacks(self, checkpoint_path):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.00001
            )
        ]
        return callbacks
        
    def train(
        self,
        x_train,
        y_train,
        validation_data=None,
        epochs=10,
        batch_size=32,
        checkpoint_path=None,
        class_weights=None,
        **kwargs
    ):
        callbacks = self.get_callbacks(checkpoint_path) if checkpoint_path else None
        
        # Handle class imbalance
        if class_weights is None and y_train is not None:
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            class_weights = {
                0: (1 / n_neg) * (len(y_train) / 2.0),
                1: (1 / n_pos) * (len(y_train) / 2.0)
            }
        
        return self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            **kwargs
        )
        
    def predict(self, x, batch_size=None):
        return self.model.predict(x, batch_size=batch_size)
    
    def predict_proba(self, x, batch_size=None):
        return self.predict(x, batch_size=batch_size)

    def save_model(self, path):
        self.model.save(path)
        
    def get_model_summary(self):
        """Get model architecture summary"""
        return self.model.summary()

    @classmethod
    def load_model(cls, path):
        """Load a saved model"""
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise 