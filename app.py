# import os
# from flask import Flask, render_template, request, redirect
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing import image
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np
# from werkzeug.utils import secure_filename

# # Setup
# app = Flask(__name__)
# model = load_model("model/crop_model.h5")  # or "model/crop_model.keras"
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Make sure the folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# classes = ['Apple Scab', 'Black Rot', 'Cedar Rust', 'Healthy']  # example classes

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     filename = None
#     if request.method == "POST":
#         file = request.files["image"]
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             # Preprocess image
#             # img = image.load_img(filepath, target_size=(224, 224))  # adjust if needed
#             # img_array = image.img_to_array(img)
#             # img_array = np.expand_dims(img_array, axis=0) / 255.0

#             img = image.load_img(filepath, target_size=(128, 128))  # <-- corrected size
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0) / 255.0


#             # Predict
#             print("Prediction raw output:", pred)
#             print("Predicted index:", np.argmax(pred))
#             print("Number of classes:", len(classes))
#             pred = model.predict(img_array)
#             prediction = classes[np.argmax(pred)]

#     return render_template("index.html", prediction=prediction, image_file=filename)
# if __name__ == "__main__":
#     app.run(debug=True)

import os
from flask import Flask, render_template, request, redirect
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Setup
app = Flask(__name__)
model = load_model("model/crop_model.keras")  # or "model/crop_model.keras"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure this matches the output layer of your model
classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Load and preprocess image (make sure size matches your model's input)
                img = image.load_img(filepath, target_size=(128, 128))  # must match training
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                pred = model.predict(img_array)
                print("Raw prediction output:", pred)
                if pred.shape[1] == len(classes):
                    prediction = classes[np.argmax(pred)]
                else:
                    prediction = "Error: model output does not match number of classes"
            except Exception as e:
                prediction = f"Error processing image: {str(e)}"

    return render_template("index.html", prediction=prediction, image_file=filename)

if __name__ == "__main__":
    app.run(debug=True)
