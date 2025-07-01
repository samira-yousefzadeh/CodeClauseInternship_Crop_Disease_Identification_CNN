import os
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Setup
app = Flask(__name__)
model = load_model("model/crop_model.h5")  # or "model/crop_model.keras"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
classes = ['Apple Scab', 'Black Rot', 'Cedar Rust', 'Healthy']  # example classes

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

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))  # adjust if needed
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            pred = model.predict(img_array)
            prediction = classes[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, image_file=filename)
