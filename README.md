# Lightweight Bee Pollination Audio Recognition
> ICAIMS 2026 IEEE Conference Project

## Project Introduction
This project uses audio signal processing and machine learning to recognize bee pollination behavior in pomelo orchards.
We extract MFCC acoustic features, then use Random Forest and SVM to classify bee audio under simulated orchard noise.

## Features
1. Load raw bee audio and generate mel spectrogram
2. Extract MFCC and audio band energy features
3. SVM / Random Forest model training and performance evaluation
4. Output confusion matrix and feature importance visualization

## Environment
- Python >= 3.8
- librosa, numpy, scikit-learn, matplotlib

Install dependencies:
```bash
pip install librosa numpy scikit-learn matplotlib
