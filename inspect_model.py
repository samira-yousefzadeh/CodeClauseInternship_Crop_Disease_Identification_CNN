from keras.models import load_model

model = load_model("model/crop_model.keras")  # or .h5
model.summary()