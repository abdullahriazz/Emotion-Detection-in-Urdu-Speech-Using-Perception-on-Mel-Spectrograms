# Emotion-Detection-in-Urdu-Speech-Using-Perception-on-Mel-Spectrograms

This repository contains the implementation of a Speech Emotion Recognition (SER) system specifically designed for the Urdu language. The project treats SER as a computer vision task by converting audio waveforms into Mel Spectrograms and analyzing them using a Band-Split Neural Network architecture based on Pretrained Audio Neural Networks (PANNs).

## Project Overview

Standard approaches to Speech Emotion Recognition often struggle with low-resource languages like Urdu due to limited datasets. This project addresses this limitation by adapting visual architectures to process audio representations.

The core model implemented in this repository is a **Band-Split PANNs Classifier**. Instead of processing the entire spectrogram uniformly, the model splits the Mel Spectrogram into frequency bands (Low, Mid, High) and applies specific attention mechanisms to each band before fusing the features for classification.

## Authors

* **Abdullah Riaz** (National University of Computer and Emerging Sciences, Lahore)
* **Zainab Khan Lodhi** (National University of Computer and Emerging Sciences, Lahore)

## Dataset

The model is trained on the **UrduSER** dataset.

* **Source:** Urdu Language Speech Dataset
* **Content:** 400 Audio Clips
* **Classes:** Angry, Happy, Neutral, Sad
* **Format:** Audio files are resampled to 32,000 Hz and standardized to a 5-second duration.

## Methodology

### 1. Preprocessing
Audio files are loaded and processed using `torchaudio`. The pipeline includes:
* **Resampling:** All audio is converted to a sample rate of 32kHz.
* **Standardization:** Clips are either padded or truncated to exactly 5 seconds.
* **Spectrogram Conversion:** Log-Mel Spectrograms are generated using a window size of 1024, hop length of 320, and 64 Mel bins.

### 2. Model Architecture
The `PANNsBandSplitClassifier` uses the **Cnn14** model (pretrained on AudioSet) as a backbone feature extractor. Key components include:
* **Backbone:** Cnn14 (from `panns_inference`) with the final layers unfrozen for fine-tuning.
* **Band Splitting:** The spectrogram is divided into Low (25%), Mid (50%), and High (25%) frequency bands.
* **Band Attention:** A custom `BandAttentionBlock` computes attention weights for each frequency band to highlight relevant spectral features.
* **Fusion:** Embeddings from the global backbone and the local band encoders are concatenated and passed through a dense classification head.

## Performance

The model achieves an accuracy of **78%** on the validation set.

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Angry | 1.00 | 0.90 | 0.95 | 20 |
| Happy | 0.57 | 0.60 | 0.59 | 20 |
| Neutral | 0.76 | 0.80 | 0.78 | 20 |
| Sad | 0.80 | 0.80 | 0.80 | 20 |
| **Overall** | | | **0.78** | **80** |

*Source: Model training outputs.*

## Installation and Usage

### Prerequisites
* Python 3.8+
* PyTorch
* CUDA-enabled GPU (Recommended)

### Dependencies
Install the required packages using pip:

```bash
pip install torch torchaudio pandas numpy scikit-learn matplotlib seaborn tqdm panns-inference
