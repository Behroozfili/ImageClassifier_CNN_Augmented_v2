import cv2
import numpy as np
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras._tf_keras.keras.utils import to_categorical
from keras import models, layers, regularizers
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from joblib import dump

# Function to process a single image (read, resize, and normalize)
def process_image(image_path, size=(32, 32)):
    img = cv2.imread(str(image_path))  # Read the image
    img = cv2.resize(img, size)  # Resize the image
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    return img

# Function to load the dataset from the specified folder
def load_dataset(dataset_folder_path):
    dataset = []  # List to store images
    labels = []   # List to store labels
    folder_path = Path(dataset_folder_path)

    # Recursively find all jpg and png files
    image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

    for i, image_path in enumerate(image_paths):
        img = process_image(image_path)  # Process the image
        dataset.append(img)  # Add image to dataset

        # Extract label from the parent folder name (class)
        label = Path(image_path).parent.name
        labels.append(label)  # Add label to labels list

        # Print processing status every 100 images
        if i % 100 == 0:
            print(f"[INFO] : {i}/{len(image_paths)} images processed")

    return np.array(dataset), np.array(labels)  # Convert to numpy arrays

# Function to preprocess data: split, encode labels, and one-hot encode
def preprocess_data(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

    # Encode string labels into numerical values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert the labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test, label_encoder

# Function to build an optimized neural network model
def build_optimized_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.BatchNormalization(),  # Adding Batch Normalization
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to plot training history (loss and accuracy)
def plot_history(history):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main execution starts here

# Load the dataset from the folder
dataset_folder_path = r"E:\machinlerning\cat_dog_neuralnetwork\Q3"
data_set, labels = load_dataset(dataset_folder_path)

# Preprocess the data (split, encode, and one-hot encode)
x_train, x_test, y_train, y_test, label_encoder = preprocess_data(data_set, labels)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             fill_mode='nearest')

# Build the model
model = build_optimized_model(input_shape=(32, 32, 3), num_classes=y_train.shape[1])

# Early Stopping and ReduceLROnPlateau to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=300,
                    validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Final Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot training history
plot_history(history)

# Save the trained model
model.save("cat_dog_optimized_v2.keras")

# Save the label encoder for future use
dump(label_encoder, r"E:\machinlerning\cat_dog_neuralnetwork\label_encoder.pkl")
