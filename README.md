
markdown
Copy
Edit
# 🐱🐶 Cat & Dog Image Classification using CNN  
**Author: Behrooz Filzadeh**

This project implements a Convolutional Neural Network (CNN) using Keras (TensorFlow backend) to classify images of cats and dogs. It covers all major steps: data loading, preprocessing, training, evaluation, and prediction. OpenCV is used for image handling, and joblib for caching and label encoding.

---

## 📚 Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
  - [1. Clone Repository](#1-clone-repository)  
  - [2. Create Virtual Environment (Recommended)](#2-create-virtual-environment-recommended)  
  - [3. Install Dependencies](#3-install-dependencies)  
  - [4. Prepare Dataset](#4-prepare-dataset)  
- [Usage](#usage)  
  - [1. Training the Model](#1-training-the-model)  
  - [2. Making Predictions](#2-making-predictions)  
- [Model Architecture](#model-architecture)  
- [Results & Evaluation](#results--evaluation)  
- [File Descriptions](#file-descriptions)  
- [Customization](#customization)  
- [License](#license)  

---

## ✅ Features

- **Data Loading & Preprocessing:**  
  - Loads images recursively from specified folders.  
  - Resizes images to a consistent size (64x64 pixels).  
  - Normalizes pixel values to [0, 1].  
  - Uses `joblib` to cache/load processed datasets for faster subsequent runs.

- **Data Augmentation:**  
  - Real-time augmentation with `ImageDataGenerator` (rotation, shift, shear, zoom, horizontal flip) to improve generalization.

- **CNN Model:**  
  - Custom architecture with Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout layers.  
  - L2 regularization applied in dense layers.

- **Training & Callbacks:**  
  - Adam optimizer with a learning rate of 0.0001.  
  - `ReduceLROnPlateau` to reduce learning rate on plateau in validation loss.  
  - `EarlyStopping` to stop training early when validation loss stops improving.

- **Evaluation:**  
  - Plots for training/validation loss and accuracy.  
  - Confusion matrix heatmap.  
  - Classification report with precision, recall, and F1-score.

- **Model Persistence:**  
  - Saves trained model in `.keras` format.  
  - Saves `LabelEncoder` using `joblib` as `.pkl` file.

- **Prediction on New Images:**  
  - Loads saved model and label encoder.  
  - Predicts classes for images in a specified folder.  
  - Annotates and displays images with predicted labels using OpenCV.

---

## 📁 Project Structure

cat-dog-classification/
├── data/
│ ├── train/ # Training images folder
│ │ ├── Cat/ # Cat images
│ │ └── Dog/ # Dog images
│ └── test_images/ # Images for prediction/testing
├── saved_models/ # Saved model and label encoder
│ ├── cat_dog_optimized_v2.keras
│ └── label_encoder.pkl
├── train_model.py # Training script
├── predict_images.py # Prediction script
├── requirements.txt # Python dependencies
└── README.md # This file

yaml
Copy
Edit

**Note:**  
The script uses an absolute path by default. Modify `dataset_folder_path` and other paths to relative paths or your local setup as needed.

---

## ⚙️ Prerequisites

- Python 3.7+  
- pip (Python package installer)  
- Git (for cloning repository)

---

## 🛠 Setup

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd cat-dog-classification
2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file with:

nginx
Copy
Edit
tensorflow
opencv-python
numpy
scikit-learn
matplotlib
seaborn
joblib
Then install:

bash
Copy
Edit
pip install -r requirements.txt
4. Prepare Dataset
Create data/train/ directory.

Inside data/train/, create Cat/ and Dog/ folders.

Place cat images inside Cat/ and dog images inside Dog/.

(Optional) Create data/test_images/ for prediction testing.

🚀 Usage
1. Training the Model
Update paths in train_model.py (e.g., dataset_folder_path = "data/train").

Update model and label encoder save paths if needed.

Run training:

bash
Copy
Edit
python train_model.py
The script will preprocess data, train the CNN, show training progress and plots, and save the model and label encoder.

2. Making Predictions
Update paths in predict_images.py for the model, label encoder, and test images folder.

Run prediction:

bash
Copy
Edit
python predict_images.py
The script will predict classes for images, annotate them, and display one by one.

🏗 Model Architecture
Input: Images resized to (64, 64, 3)

Conv Block 1: Conv2D(32 filters, 3x3), ReLU, BatchNorm, MaxPooling(2x2)

Conv Block 2: Conv2D(64 filters, 3x3), ReLU, BatchNorm, MaxPooling(2x2)

Conv Block 3: Conv2D(128 filters, 3x3), ReLU, BatchNorm, MaxPooling(2x2)

Flatten

Dense(256 units, ReLU, L2 regularization)

Dropout(0.5)

BatchNormalization

Output Dense layer with softmax activation (number of classes)

Optimizer: Adam (lr=0.0001)
Loss: Categorical Cross-Entropy

📊 Results & Evaluation
Training and validation loss/accuracy curves plotted.

Confusion matrix heatmap displayed.

Classification report printed with precision, recall, F1-score, and support.

Final test accuracy and loss printed in console.

📂 File Descriptions
train_model.py: Loads data, trains model, evaluates, saves model and label encoder.

predict_images.py: Loads model and encoder, predicts on new images, displays results.

requirements.txt: Python package dependencies.

dataset_processed.joblib: Cached processed dataset (generated by train_model.py).

saved_models/cat_dog_optimized_v2.keras: Saved Keras model file.

saved_models/label_encoder.pkl: Saved LabelEncoder file.

🎨 Customization
Change image size in process_image and model input shape.

Adjust model layers, units, filters, activations in build_simple_model.

Tune hyperparameters like learning rate, batch size, epochs, dropout rate, regularization strength.

Modify data augmentation parameters in ImageDataGenerator.

Update dataset and save paths to match your environment.

