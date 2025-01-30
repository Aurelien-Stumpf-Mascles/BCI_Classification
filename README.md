
# BCI Classification

This repository contains code and resources for classifying EEG signals in Brain-Computer Interface (BCI) applications. The project focuses on implementing and evaluating various machine learning models to accurately interpret EEG data.

## Overview

Brain-Computer Interfaces enable direct communication between the brain and external devices by analyzing brain signals, particularly EEG data. This project aims to classify motor imagery EEG signals using different neural network architectures.

## Repository Structure

- `eeg_project_package/`: Includes implementations of datasets structures, training and various neural network architectures
- `experiments/`: Jupyter notebooks demonstrating data preprocessing, model training, and evaluation.
- `test/`: Python scripts which explore diverse methods
- `README.md`: Project overview and instructions.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aurelien-Stumpf-Mascles/BCI_Classification.git
   cd BCI_Classification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Jupyter Notebooks**: Explore the `notebooks/` directory for detailed walkthroughs and analyses.

## Models Implemented

- **EEGNet**: A compact convolutional neural network for EEG-based BCI applications.
- **DeepConvNet**: A deep convolutional network tailored for EEG signal classification.

## Results

Evaluation metrics and results for each model are stored in the `results/` directory. Detailed analyses can be found in the Jupyter notebooks.

## References

- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/1611.08024)
- [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization](https://arxiv.org/abs/1703.05051)
