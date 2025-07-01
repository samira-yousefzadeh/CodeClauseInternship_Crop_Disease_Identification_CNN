import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

# === Setup ===
model_path = "model/crop_model.keras"
image_path = "patato.jpeg"

# === Class Labels (ensure this matches model's output layer) ===
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# === Load Model ===
print("Loading model...")
model = load_model(model_path)
print("Model loaded.\n")

# === Load and Preprocess Image ===
print(f"Loading image: {image_path}")
img = image.load_img(image_path, target_size=(128, 128))  # resize must match training
img_array = image.img_to_array(img)

print("Raw image shape:", img_array.shape)  # Should be (128, 128, 3)

img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
print("Input shape to model:", img_array.shape)  # Should be (1, 128, 128, 3)

# === Prediction ===
pred = model.predict(img_array)
print("Raw prediction output:", pred)
print("Prediction shape:", pred.shape)

# === Decode Prediction ===
if pred.shape[1] == len(classes):
    predicted_class = classes[np.argmax(pred)]
    print("✅ Predicted Class:", predicted_class)
else:
    print("❌ Error: model output does not match number of classes.")
