import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image
import pathlib

# Load the saved model
model = tf.keras.models.load_model('fursuit_classifier.h5')

# Define image preprocessing function
def preprocess_image(img_path, img_height=64, img_width=64):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Define class names and their corresponding labels
data_dir = pathlib.Path(__file__).parent / 'TP2-images'
class_names = sorted([item.name for item in data_dir.glob('*')])

# Define fursuit class and mapping for animals
fursuit_class_name = "fursuit"  # The exact name should match one of the class_names

# Map Italian class names to French
class_name_translation = {
    "cane": "chien",
    "elefante": "elephant",
    "cavallo": "cheval",
    "farfalla": "papillon",
    "gallina": "poulet",
    "gatto": "chat",
    "mucca": "vache",
    "pecora": "mouton",
    "ragno": "araignee",
    "scoiattolo": "ecureuil"
}

# Reverse the mapping for easier lookup
french_to_italian = {v: k for k, v in class_name_translation.items()}

# Classify a new image
def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    
    # Translate the class name
    if predicted_class == fursuit_class_name:
        print("Fursuit: Yes")
        print("Espece: N/A")
    else:
        french_class = class_name_translation.get(predicted_class, "Unknown")
        print("Fursuit: No")
        print(f"Espece: {french_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file to classify')
    args = parser.parse_args()

    classify_image(args.image_path)
