import cv2
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from keras._tf_keras.keras.utils import to_categorical
from keras import models, layers, regularizers
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from joblib import dump, load

# Function to process a single image (read, resize, and normalize)
def process_image(image_path, size=(64, 64)):
    try:
        img = cv2.imread(str(image_path))  
        if img is None:
            raise ValueError(f"Could not read the image: {image_path}")
        img = cv2.resize(img, size)  
        img = img / 255.0  # Normalize pixel values to the range [0, 1]
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def load_dataset(dataset_folder_path, save_path="dataset_processed.joblib"):
    try:
        # Check if the processed data already exists
        if os.path.exists(save_path):
            print("[INFO] : Loading processed dataset from file...")
            data = load(save_path)  # Load processed data from file
            return data['dataset'], data['labels']

        dataset = []  # List to store images
        labels = []   # List to store labels
        folder_path = Path(dataset_folder_path)

        # Recursively find all jpg and png files
        image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

        if not image_paths:
            raise FileNotFoundError("No images found in the specified folder.")

        for i, image_path in enumerate(image_paths):
            img = process_image(image_path)  
            if img is not None:
                dataset.append(img)  
                # Extract label from the parent folder name (class)
                label = Path(image_path).parent.name
                labels.append(label)  
                # Print processing status every 100 images
                if i % 100 == 0:
                    print(f"[INFO] : {i}/{len(image_paths)} images processed")
            else:
                print(f"[WARNING] Skipped image: {image_path}")

        if len(dataset) == 0:
            raise ValueError("No valid images found in the dataset folder.")
        
        print("[INFO] : Saving processed dataset to file...")
        dump({'dataset': np.array(dataset), 'labels': np.array(labels)}, save_path)

        return np.array(dataset), np.array(labels)  
    
    except FileNotFoundError as fnf_error:
        print(f"[ERROR] : {fnf_error}")
        return None, None
    except Exception as e:
        print(f"[ERROR] : Error loading dataset: {e}")
        return None, None


def preprocess_data(data, labels):
    try:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, x_test, y_train, y_test, label_encoder

    except Exception as e:
        print(f"[ERROR] : Error in data preprocessing: {e}")
        return None, None, None, None, None

# Function to build an optimized neural network model
def build_simple_model(input_shape, num_classes):
    try:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        return model

    except Exception as e:
        print(f"[ERROR] : Error building the model: {e}")
        return None


def plot_history(history):
    try:
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
    except Exception as e:
        print(f"[ERROR] : Error plotting history: {e}")

def plot_confusion_matrix(y_true, y_pred, classes):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
    except Exception as e:
        print(f"[ERROR] : Error plotting confusion matrix: {e}")

# Main execution
if __name__ == "__main__":
    dataset_folder_path = r"E:\machinlerning\cat_dog_neuralnetwork\Q3"
    
    # Load dataset
    data_set, labels = load_dataset(dataset_folder_path)
    if data_set is None or labels is None:
        print("[ERROR] : Failed to load the dataset. Exiting.")
        exit()

    # Preprocess data
    x_train, x_test, y_train, y_test, label_encoder = preprocess_data(data_set, labels)
    if x_train is None or y_train is None:
        print("[ERROR] : Failed to preprocess the data. Exiting.")
        exit()

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                 fill_mode='nearest')

    # Build model
    model = build_simple_model(input_shape=(64, 64, 3), num_classes=y_train.shape[1])
    if model is None:
        print("[ERROR] : Model building failed. Exiting.")
        exit()

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)

    # Train the model
    try:
        history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=300,
                            validation_data=(x_test, y_test), callbacks=[reduce_lr, early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Final Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Plot history and confusion matrix
        plot_history(history)
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)) 
        plot_confusion_matrix(y_true, y_pred_classes, classes=label_encoder.classes_) 

        # Save model and label encoder
        model.save("cat_dog_optimized_v2.keras")
        dump(label_encoder, r"E:\machinlerning\cat_dog_neuralnetwork\label_encoder.pkl")
    except Exception as e:
        print(f"[ERROR] : Error during model training or evaluation: {e}")
