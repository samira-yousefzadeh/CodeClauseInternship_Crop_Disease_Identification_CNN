
# PlantVillage Disease Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify images of plant leaves into different disease categories using the **PlantVillage dataset**. It was developed as part of a hands-on AI internship project and runs on **Google Colab** using TensorFlow and TensorFlow Datasets (TFDS).  
The trained model is also deployed in a simple **Flask-based web interface** for real-time image classification.

## Project Structure

- `PlantVillage_CNN_Colab_CLEAN.ipynb` – Notebook for training the CNN on PlantVillage dataset.
- `app.py` – Flask web application for uploading and classifying plant images.
- `test.py` – Contains helper functions for preprocessing and predicting uploaded images.
- `inspect_model.py` – Utility for inspecting the trained model structure and predictions.
- `crop_model.keras` – Trained Keras model file used in the web interface.
- `kaggle.json` – API token file for accessing Kaggle datasets.
- `README.md` – Project overview and instructions (you are here).

## 📊 Dataset

The project uses the [PlantVillage dataset](https://www.tensorflow.org/datasets/catalog/plant_village) via `tensorflow_datasets`. It contains over 54,000 labeled images of healthy and diseased plant leaves from 38 different classes.

## Model Architecture

- Input shape: resized RGB images (e.g., 180x180)
- Layers:
  - Convolution + MaxPooling
  - Dropout
  - Dense
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

## How to Run

### Train the Model
1. Open the `PlantVillage_CNN_Colab_CLEAN.ipynb` notebook.
2. Run all cells (requires TensorFlow and TFDS).
3. Save the trained model as `crop_model.keras`.

### 🌐 Launch the Flask Web App
1. Install dependencies:
```bash
pip install flask tensorflow pillow
```

2. Run the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to `http://127.0.0.1:5000` to use the interface.

4. Upload a plant image to classify.

## Web Interface Preview

The Flask app allows users to:
- Upload an image of a plant leaf.
- Run the trained CNN model on the image.
- Get a prediction of the disease (or healthy status) directly in the browser.

## ✅ Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask
- Pillow (for image processing)

Install with:
```bash
pip install tensorflow flask pillow
```

## 📚 License

This project is open source and available under the [MIT License](LICENSE).
