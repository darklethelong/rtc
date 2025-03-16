# Financial Customer Complaint Detection System - Implementation Guide

## Project Overview
This system uses a hybrid CNN-RNN architecture with attention mechanism to detect complaints in customer service conversations for financial institutions. The model is specifically designed to handle class imbalance and long text sequences in banking domain conversations.

## Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (optional, but recommended for faster training)

## Step 1: Environment Setup

### Clone the Repository
```bash
git clone https://github.com/darklethelong/rtc.git
cd rtc
git checkout dev1
```

### Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Required NLTK Data
```bash
python download_nltk_data.py
```

## Step 2: Project Structure
Ensure your project has the following structure:
```
complaint_detection/
├── data/
│   └── raw/
│       └── conversations.csv
├── models/
│   ├── __init__.py
│   ├── complaint_classifier.py
│   └── saved_models/
├── src/
│   ├── __init__.py
│   ├── run_training.py
│   └── train.py
└── utils/
    ├── __init__.py
    └── preprocessor.py
```

## Step 3: Data Preparation

### Data Format
Your `conversations.csv` should have the following format:
```csv
text,label
"conversation text here",complaint
"another conversation text",non-complaint
```

### Data Requirements
- Text column: Contains the conversation text (supports up to 1000 characters)
- Label column: Contains either "complaint" or "non-complaint"
- Recommended minimum: 1000 samples
- Current support for imbalanced data: 1500 complaint vs 5000 non-complaint samples

## Step 4: Model Configuration

### Adjust Hyperparameters (if needed)
In `complaint_detection/models/complaint_classifier.py`:
```python
class ComplaintClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,     # Adjust for text complexity
        hidden_dim=128,        # Adjust for model capacity
        num_filters=128,       # Adjust for feature extraction
        filter_sizes=[3, 4, 5, 6],  # Adjust for n-gram capture
        dropout=0.4,           # Adjust for regularization
    ):
```

### Modify Domain Knowledge (if needed)
In `complaint_detection/utils/preprocessor.py`, update the domain-specific terms:
- Banking terms
- Complaint indicators
- Service context
- Stop words

## Step 5: Training the Model

### Basic Training
```bash
python complaint_detection/src/run_training.py
```

### Advanced Training Options
Modify in `complaint_detection/src/run_training.py`:
```python
model, preprocessor, history = train_model(
    data_path=data_path,
    model_save_path=model_save_path,
    preprocessor=preprocessor,
    epochs=30,          # Adjust training duration
    batch_size=32,      # Adjust based on GPU memory
    learning_rate=0.001,
    class_weights=class_weights
)
```

## Step 6: Model Evaluation
The training process automatically:
1. Saves the best model based on validation loss
2. Generates ROC and PR curves
3. Provides classification metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - AUC-ROC

## Step 7: Making Predictions
```python
from complaint_detection.models.complaint_classifier import ComplaintClassifier
from complaint_detection.utils.preprocessor import TextPreprocessor

# Load saved model and preprocessor
model = ComplaintClassifier.load_model('path/to/saved_model.pt')
preprocessor = TextPreprocessor.load('path/to/preprocessor.json')

# Preprocess new text
processed_text = preprocessor.transform_texts([your_text])

# Get predictions
predictions, probabilities = model.predict(processed_text, device='cpu')
```

## Performance Optimization Tips

### For Large Datasets
1. Increase batch size if GPU memory allows
2. Enable multi-GPU training if available
3. Use gradient accumulation for very large batches

### For Imbalanced Data
1. Adjust class weights in `run_training.py`
2. Consider data augmentation for minority class
3. Experiment with different sampling strategies

### For Long Sequences
1. Adjust max_len in TextPreprocessor
2. Modify CNN filter sizes for better feature capture
3. Consider using hierarchical attention

## Troubleshooting

### Common Issues
1. Out of Memory (OOM):
   - Reduce batch size
   - Reduce model dimensions
   - Use gradient checkpointing

2. Poor Performance:
   - Check class distribution
   - Verify domain knowledge coverage
   - Adjust model hyperparameters

3. Slow Training:
   - Enable GPU training
   - Optimize batch size
   - Check data loading bottlenecks

## Monitoring and Logging
- Training metrics are logged to console
- Model checkpoints are saved in `saved_models/`
- Visualization plots are generated automatically

## Best Practices
1. Always version your data and models
2. Monitor validation metrics for overfitting
3. Regularly update domain knowledge
4. Test model robustness with diverse inputs
5. Implement proper error handling
6. Keep preprocessor and model in sync

## Support and Updates
For issues or improvements:
1. Check the GitHub repository
2. Submit detailed bug reports
3. Contribute to domain knowledge
4. Share performance metrics

## Security Considerations
1. Sanitize input data
2. Secure model deployment
3. Handle sensitive financial information appropriately
4. Implement proper access controls 