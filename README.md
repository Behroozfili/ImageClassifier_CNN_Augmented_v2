🐱🐶 Cat & Dog Image Classification using CNN
Author: Behrooz Filzadeh

This project implements a Convolutional Neural Network (CNN) using Keras (TensorFlow backend) to classify images of cats and dogs. It covers all major steps: data loading, preprocessing, training, evaluation, and prediction. OpenCV is used for image handling, and joblib for caching and label encoding.

📚 Table of Contents
Features

Project Structure

Prerequisites

Setup

Usage

Model Architecture

Results & Evaluation

File Descriptions

Customization

License

✅ Features
Data Handling

Recursive image loading from subfolders

Resizes images to 64x64 pixels

Normalizes pixel values (0–1 range)

Caches processed dataset using joblib for faster re-runs

Data Augmentation

Real-time augmentation using ImageDataGenerator (rotation, shifts, zoom, flip, etc.)

CNN Architecture

Custom CNN with Conv2D, MaxPooling, BatchNormalization, Dropout, and Dense layers

Includes L2 regularization

Training Utilities

Optimizer: Adam with learning rate scheduling

EarlyStopping and ReduceLROnPlateau callbacks

Evaluation Tools

Plots for accuracy/loss

Confusion matrix heatmap

Precision, recall, and F1-score report

Persistence

Saves trained model (.keras) and label encoder (.pkl)

Predicts on new images with annotated output via OpenCV

📁 Project Structure
kotlin
Copy
Edit
cat-dog-classification/
├── data/
│   ├── train/
│   │   ├── Cat/
│   │   └── Dog/
│   └── test_images/
├── saved_models/
│   ├── cat_dog_optimized_v2.keras
│   └── label_encoder.pkl
├── train_model.py
├── predict_images.py
├── requirements.txt
└── README.md
⚠️ Note: Update any hardcoded absolute paths like E:\machinlerning\... to use relative paths for better portability.

💻 Prerequisites
Python 3.7+

pip (Python package manager)

Git

⚙️ Setup
1. Clone the Repository
bash
Copy
Edit
git clone <your-repo-url>
cd cat-dog-classification
2. Create a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
3. Install Dependencies
First, ensure requirements.txt contains:

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
Create folders:

bash
Copy
Edit
data/train/Cat/
data/train/Dog/
data/test_images/
Put cat and dog images into their respective folders.

🚀 Usage
1. Train the Model
Edit train_model.py and set:

python
Copy
Edit
dataset_folder_path = "data/train"
model.save("saved_models/cat_dog_optimized_v2.keras")
dump(label_encoder, "saved_models/label_encoder.pkl")
Then run:

bash
Copy
Edit
python train_model.py
The script will preprocess and cache the data, train the model, and save the model and label encoder.

2. Predict New Images
Edit predict_images.py:

python
Copy
Edit
model_path = "saved_models/cat_dog_optimized_v2.keras"
label_encoder_path = "saved_models/label_encoder.pkl"
test_images_folder_path = "data/test_images"
Then run:

bash
Copy
Edit
python predict_images.py
The script will annotate and display predicted labels on your test images.

🧠 Model Architecture
Input: (64, 64, 3)
Layers:

Conv2D (32 filters) + BatchNorm + MaxPooling

Conv2D (64 filters) + BatchNorm + MaxPooling

Conv2D (128 filters) + BatchNorm + MaxPooling

Flatten

Dense (256 units, ReLU, L2) + Dropout (0.5) + BatchNorm

Output Dense (Softmax for classification)

Compiled with:
Adam(learning_rate=0.0001) + categorical_crossentropy

📊 Results & Evaluation
After training, you’ll get:

Accuracy/loss plots (matplotlib)

Confusion matrix (seaborn heatmap)

Classification report (scikit-learn)

Final test accuracy printed in the terminal

📄 File Descriptions
File	Description
train_model.py	Loads and trains the CNN, saves model & encoder
predict_images.py	Predicts and displays labeled images
requirements.txt	Python dependencies
dataset_processed.joblib	Cached processed images & labels
saved_models/	Stores model (.keras) & encoder (.pkl)

🔧 Customization
Option	How to Modify
Image size	In process_image() and CNN input layer
Model layers	Edit build_simple_model()
Augmentation	Change ImageDataGenerator settings
Hyperparameters	Adjust learning rate, dropout, epochs, etc.
Dataset paths	Update dataset_folder_path and related paths

