## Person Identification and Classification Deep Learning Project

This repository contains a deep learning system for person identification and classification, comparing two models: a baseline CNN model and a Local Derivative Pattern (LDP) model. Both models take images of celebrities as input and predict their names.

## Overview
The project explores and compares the effectiveness of two different approaches to person identification using deep learning techniques:

1) Baseline CNN Model:
- Utilizes a Convolutional Neural Network (CNN) architecture for image classification.
- Trained on a dataset of celebrity images to predict the names of celebrities.

2) Local Derivative Pattern (LDP) Model:
- Implements a model with Local Derivative Pattern for feature extraction.
- Trained on the same celebrity dataset as the baseline CNN model.
- Predicts the names of celebrities based on local patterns in the images.

# Dataset
The dataset used in this project includes images of celebrities obtained from the Labeled Faces in the Wild (LFW) dataset. The dataset is divided into training and testing sets.

# Requirements
To run the code, ensure you have the required dependencies installed. 
You can install them using the following command:
pip install -r requirements.txt

# Training Models
1) Baseline CNN Model:
python train_baseline_model.py

2) LDP Model:
python train_ldp_model.py

# Evaluating Models
Evaluate the models and compare their performance:
python evaluate_models.py

# Results
- The results and performance metrics of both models are documented in the results directory. 
- Visualization of accuracy and loss over epochs is available in the generated plots.

# Model Comparison
The project includes visualizations comparing the training and validation accuracy as well as loss for both the LDP and baseline CNN models.

# Inference
- The saved models (ldp_model.h5 and baseline_model.h5) can be used for inference on new images. 
- The code includes an example of using the models to make predictions on selected test images.

# Conclusion
- The README provides an overview of the project structure, data processing, model training, and evaluation. 
- It aims to assist users in understanding and reproducing the experiments conducted in the project.

# License
This project is licensed under the MIT License.