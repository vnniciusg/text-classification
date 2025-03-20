# IMDB Reviews Text Classification

A deep learning-based text classification system for sentiment analysis on the IMDB movie reviews dataset using TensorFlow and Bidirectional LSTM networks.

## Overview

This project implements a Recurrent Neural Network (RNN) for classifying IMDB movie reviews as either positive or negative. The implementation uses TensorFlow and Keras to build a bidirectional LSTM architecture that can effectively capture the sentiment in review texts.

## Project Structure

- **config/**: Configuration settings for the model and training process
- **data/**: Data loading and preprocessing logic
- **model/**: RNN text classifier model implementation
- **utils/**: Utility functions including performance timing decorator

## Features

- Bidirectional LSTM architecture for capturing context in both directions
- TextVectorization layer for preprocessing text data
- Configurable training parameters via Pydantic settings
- Performance monitoring with timing decorators

## Requirements

- Python 3.11+
- TensorFlow
- Loguru for logging
- Pydantic for settings management

## Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/vnniciusg/text-classification.git
cd text-classification
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the classifier**

```bash
python main.py
```

## Configuration

You can modify the training parameters in `config/__init__.py`:

- `BUFFER_SIZE`: Size of the shuffle buffer (default: 10000)
- `BATCH_SIZE`: Batch size for training (default: 64)
- `VOCAB_SIZE`: Maximum vocabulary size (default: 1000)
- `TRAINING_LEARNING_RATE`: Learning rate for Adam optimizer (default: 1e-4)
- `EPOCHS`: Number of training epochs (default: 10)

## Model Architecture

The model consists of:

- Text vectorization layer
- Embedding layer
- Two Bidirectional LSTM layers
- Dense layers for classification

## License

[MIT License](LICENSE)
