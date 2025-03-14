# Real-Time Call Center Complaint Detection System

A machine learning system for real-time detection of customer complaints in call center conversations.

## Overview

This project implements an advanced NLP system for detecting customer complaints in call center conversations as they happen in real-time. The system analyzes the context of conversations through a sliding window approach and produces a real-time visualization of complaint probability, similar to a stock price chart.

### Key Features

- **Real-time complaint detection**: Analyzes utterances as they happen
- **Context-aware**: Uses a window of recent utterances for better detection
- **Hybrid CNN-LSTM architecture**: Combines convolutional and recurrent neural networks for optimal text processing
- **Real-time visualization**: Dynamic charting of complaint probability over time
- **Handles speech-to-text artifacts**: Robust to transcription errors and disfluencies

## Project Structure

```
complaint_detection/
├── data/                # Data storage
├── models/              # Saved models
├── src/                 # Source code
│   ├── data_preprocessing.py     # Text preprocessing
│   ├── model.py                  # CNN-LSTM model architecture
│   ├── train.py                  # Model training script
│   └── real_time_detection.py    # Real-time visualization dashboard
├── utils/               # Utility functions
├── visualization/       # Visualization outputs
├── requirements.txt     # Project dependencies
├── run.py               # Main runner script
└── README.md            # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Streamlit
- NLTK
- Pandas, NumPy
- Plotly
- Scikit-learn
- Additional dependencies in requirements.txt

## Installation

1. Clone this repository:

```bash
git clone https://github.com/darklethelong/rtc.git
cd rtc
```

2. Install dependencies:

```bash
# Option 1: Using pip directly
pip install -r requirements.txt

# Option 2: Using the run script
python run.py install
```

## Usage

### 1. Preparing Your Data

Format your data as a CSV file with at least two columns:
- `text`: The conversation text, with speakers indicated by "Agent:" and "Caller:" prefixes
- `label`: The label for the conversation ("complaint" or "non-complaint")

Example:
```
text,label
"Agent: How can I help you today?
Caller: I've been charged twice for my subscription.
Agent: I apologize for that. Let me check your account.",complaint
```

### 2. Training the Model

```bash
python run.py train --data_path <path_to_your_data.csv> --output_dir output --epochs 10
```

Additional training options:
```
--window_size INT        Number of utterances in each window (default: 3)
--embedding_dim INT      Dimension of embedding vectors (default: 100)
--max_sequence_length INT Maximum sequence length (default: A128)
--epochs INT            Number of training epochs (default: 10)
--batch_size INT        Training batch size (default: 32)
--use_class_weights     Use class weights for imbalanced data
```

### 3. Running the Real-Time Dashboard

```bash
python run.py dashboard
```

This will start a Streamlit dashboard where you can input conversation utterances and see real-time complaint detection.

## Model Architecture

The system uses a hybrid CNN-LSTM architecture:

1. **Text Processing Branch**:
   - Embedding layer to convert text to vectors
   - Multiple convolutional layers for n-gram pattern detection
   - Bidirectional LSTM layer for sequential context

2. **Feature Processing Branch**:
   - Processes conversation metadata and extracted features
   - Includes speaker ratio, question counts, etc.

3. **Combined Neural Network**:
   - Merges text and feature branches
   - Dense layers for final classification
   - Sigmoid output for complaint probability

## Performance Optimization

The model is optimized for real-time performance through:

1. Efficient preprocessing pipeline with caching
2. Lightweight CNN filters for fast inference
3. Minimal LSTM units to reduce computational complexity
4. Batch normalization for faster convergence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was built to improve call center customer experience
- Special thanks to all contributors 