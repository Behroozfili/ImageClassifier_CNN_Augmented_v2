import cv2
import numpy as np
import os
import glob
from keras._tf_keras.keras.models import load_model
from joblib import load

def load_and_preprocess_image(image_path, size=(64, 64)):
    """Load and preprocess a single image for prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize pixel values
    return img

def predict_and_annotate_images(model, label_encoder, image_folder_path):
    """Predict classes for images in the specified folder and annotate them."""
    # Updated glob pattern to include more extensions and be case insensitive
    image_paths = glob.glob(os.path.join(image_folder_path, '**', '*.[JjPp][PpNn]*[Gg]'), recursive=True)

    if not image_paths:
        print("[ERROR]: No images found in the specified folder.")
        return

    for image_path in image_paths:
        try:
            img = load_and_preprocess_image(image_path)  # Load and preprocess the image
            img_input = np.expand_dims(img, axis=0)  # Add batch dimension
            predictions = model.predict(img_input)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            # Load the original image for displaying results
            original_img = cv2.imread(image_path)
            # Annotate the image with the predicted label
            cv2.putText(original_img, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color text
            
            # Display the annotated image
            cv2.imshow("Predicted Image", original_img)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"[ERROR]: Error processing image {image_path}: {e}")

# Main execution for testing
if __name__ == "__main__":
    model_path = "cat_dog_optimized_v2.keras"  # Path to the saved model
    label_encoder_path = r"E:\machinlerning\cat_dog_neuralnetwork\label_encoder.pkl"  # Path to the label encoder
    test_images_folder_path = r"E:\machinlerning\cat_dog_neuralnetwork\Test\Cat" # Folder containing test images

    # Load the trained model and label encoder
    model = load_model(model_path)
    label_encoder = load(label_encoder_path)

    # Predict and annotate images
    predict_and_annotate_images(model, label_encoder, test_images_folder_path)
