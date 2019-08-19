import os

from keras.models import load_model


def load_trained_model(model_path: str):
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("model loaded successfully")
        return model
    else:
        print("unable to load model")
