# Student AI - Dropout Prediction

A machine learning project for predicting student dropout using trained models.

## Overview

This project contains pre-trained models for student dropout prediction:
- `dropout_model.pkl`: The trained dropout prediction model
- `scaler.pkl`: Data scaler for preprocessing input features

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To use the trained models, load them using Python:

```python
import pickle

# Load the model and scaler
with open('dropout_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use the model for predictions
# predictions = model.predict(scaled_data)
```

## Model Information

- Model Type: Dropout prediction model
- Preprocessing: StandardScaler (contained in scaler.pkl)

## License

See LICENSE file for details.