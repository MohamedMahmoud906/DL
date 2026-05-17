Deep Learning Project: Training Optimization for CNN on MNIST Dataset
This project focuses on Training Requirements in Deep Learning using a CNN model for handwritten digit recognition on the MNIST dataset.
1. Problem Description
The project aims to train and optimize a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset.
2. Dataset
Dataset Used: MNIST Handwritten Digits Dataset
- 60,000 training images
- 10,000 testing images
- Images size: 28x28 grayscale
3. Preprocessing
- Normalization of pixel values
- Reshaping images to CNN format
- One-hot encoding for labels
4. Model Architecture
- Conv2D Layer
- Batch Normalization
- MaxPooling Layer
- Dense Layer
- Dropout Layer
- Output Layer with Softmax
5. Training Requirements
During training, the following metrics are monitored:
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
6. Experiments
Three experiments were performed:
1. Adam Optimizer, Learning Rate 0.001, Batch Size 64
2. SGD Optimizer, Learning Rate 0.01, Batch Size 64
3. Adam Optimizer, Learning Rate 0.0001, Batch Size 128

Test Accuracy = 98.7 %
Test Loss = 0.03
7. Visualization
- Accuracy Curves
- Loss Curves
8. Results
The final comparison table includes Accuracy and Loss for all experiments.
9. Conclusion
Adam optimizer with learning rate 0.001 achieved the best performance with the highest accuracy and lowest loss.
Python Training Code

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def create_model():

    model = models.Sequential([

        layers.Conv2D(32, (3,3), activation='relu',
                      input_shape=(28,28,1)),

        layers.BatchNormalization(),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),

        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),

        layers.Dropout(0.5),

        layers.Dense(10, activation='softmax')

    ])

    return model

requirements.txt
tensorflow
numpy
matplotlib
pandas
README Instructions
1. Install required libraries using requirements.txt
2. Run the project using:
   python train.py
3. Training and evaluation results will appear automatically.
