# Real-Time Call Center Complaint Detection System

A real-time system for monitoring and analyzing customer complaints in call center conversations using deep learning.

## Overview

This project implements an advanced NLP system for detecting customer complaints in call center conversations as they happen in real-time. The system analyzes the context of conversations through a sliding window approach and produces a real-time visualization of complaint probability, similar to a stock price chart.

## Key Features

- Real-time complaint detection using hybrid CNN-LSTM model
- Interactive visualization of complaint probabilities
- Multi-level complaint classification (High, Moderate, Mild, None)
- Context-aware analysis using sliding window approach
- Support for timestamped conversations
- Session recording and playback
- Comprehensive performance metrics
- Handles speech-to-text artifacts and transcription errors

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/darklethelong/rtc.git
cd rtc
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python src/train.py --data_path data/conversations.csv
```

5. Start monitoring:
```bash
python src/monitor.py --model_path models/saved_models/model_latest.h5
```

## Project Structure

```
complaint_detection/
├── data/                # Data storage
├── models/              # Saved models and checkpoints
├── src/                 # Source code
│   ├── data_preprocessing.py     # Text preprocessing
│   ├── model.py                  # CNN-LSTM model architecture
│   ├── train.py                  # Model training script
│   └── real_time_detection.py    # Real-time visualization dashboard
├── utils/               # Utility functions
├── tests/              # Unit tests
├── notebooks/          # Development notebooks
├── visualization/      # Visualization outputs
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Model Architecture

The system uses a hybrid architecture combining:
- CNN layers for n-gram feature extraction
- Bidirectional LSTM for sequential understanding
- Multi-head attention mechanism
- Residual connections and batch normalization

### Architecture Details

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

## Real-time Monitoring

The monitoring system provides:
1. Probability Timeline
   - Real-time complaint probability tracking
   - Color-coded threshold indicators
   - Sliding window view

2. Current Status
   - Gauge visualization
   - Complaint level indicator
   - Probability score

3. Distribution Analysis
   - Complaint level distribution
   - Historical trends
   - Count statistics

## Performance

The model achieves:
- Accuracy: ~90% on test set
- F1-Score: ~0.88
- Real-time processing: <100ms per utterance

### Performance Optimization

The model is optimized for real-time performance through:
1. Efficient preprocessing pipeline with caching
2. Lightweight CNN filters for fast inference
3. Minimal LSTM units to reduce computational complexity
4. Batch normalization for faster convergence

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Special thanks to the TensorFlow team for their excellent framework
- Inspired by various research papers in the field of conversation analysis
- Built to improve call center customer experience
