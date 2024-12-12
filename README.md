# CIFAR-10 Image Classification with Convolutional Neural Networks (CNN)

## Overview

In this project, we implement a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10** dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. These classes are:

- **Airplane**
- **Automobile**
- **Bird**
- **Cat**
- **Deer**
- **Dog**
- **Frog**
- **Horse**
- **Ship**
- **Truck**

The goal of this model is to classify images into one of these 10 categories. We train a CNN model to learn patterns from these images and predict the corresponding class. The model is evaluated based on its accuracy, and the predictions are displayed along with their associated confidence levels.

By using techniques such as **Convolutional Layers**, **Max-Pooling**, and **Batch Normalization**, the model is able to classify images with high accuracy.

---

## Key Steps

### 1. **Data Preprocessing**
We load and normalize the CIFAR-10 dataset for training and testing. The dataset is split into 50,000 training images and 10,000 test images.

### 2. **Model Architecture**
A deep CNN model is designed for image classification. The architecture includes:
- Multiple convolutional layers
- Max-pooling layers
- Batch normalization
- Fully connected layers with softmax activation

### 3. **Model Training and Evaluation**
The model is trained for **40 epochs** and evaluated based on accuracy and prediction performance.

---

## Results

The model was trained for **40 epochs** on the CIFAR-10 dataset, and the final results are as follows:

- **Training Accuracy**: 94.42%
- **Training Loss**: 0.1734
- **Validation Accuracy**: 80.54%
- **Validation Loss**: 0.9290

### Observations:
- The **training accuracy** steadily increased throughout the training process, reaching 94.42% by the final epoch.
- The **validation accuracy** also showed improvement, but with a noticeable gap compared to the training accuracy. This suggests that the model may be overfitting to the training data.
- The **training loss** decreased consistently, while the **validation loss** showed some fluctuations, indicating that the model could potentially benefit from further adjustments to prevent overfitting.

---

## Model Deployment

The trained model is deployed on Hugging Face using **Streamlit**. You can access and interact with the deployed application through the following link:

[**Hugging Face Space - Image Classification for CIFAR-10**](https://huggingface.co/spaces/Senasu/Image_Classification_for_CIFAR10)

---

## Dataset

The CIFAR-10 dataset is available for download from Kaggle. You can access the dataset through the following link:

[**CIFAR-10 Dataset on Kaggle**](https://www.kaggle.com/competitions/cifar-10)
