# 👗 FashionMNIST Classification Using CNN in PyTorch

This project implements a convolutional neural network (CNN) in PyTorch to classify images from the FashionMNIST dataset. It includes data loading, preprocessing, model definition, training, and evaluation.

## 📂 Dataset

The dataset used is the **FashionMNIST dataset**, containing grayscale images of 10 categories of clothing items.

- Images are 28x28 pixels, grayscale.
- 60,000 training samples and 10,000 test samples.
- Easily accessible via `torchvision.datasets.FashionMNIST`.

## 🧠 Model Architecture

The CNN model consists of:

- Conv2d: 1 input channel → 8 output channels, kernel size 3, padding 1
- BatchNorm2d and ReLU
- MaxPool2d (kernel size 2)
- Conv2d: 8 → 32 output channels, kernel size 5, padding 2
- BatchNorm2d and ReLU
- MaxPool2d (kernel size 2)
- Fully connected layers:
  - Linear (7*7*32 → 600)
  - ReLU and Dropout (p=0.5)
  - Linear (600 → 10 classes)

## ⚙️ Training Setup

- Loss: CrossEntropyLoss
- Optimizer: Adam with learning rate 0.01
- Batch size: 100
- Epochs: 5
- Runs on GPU if available, otherwise CPU

## 📈 Usage

- Training and testing loss and accuracy are printed after each epoch.
- You can run the training script directly after installing dependencies.

## 🔧 Requirements

- Python 3.x
- torch
- torchvision
- matplotlib
