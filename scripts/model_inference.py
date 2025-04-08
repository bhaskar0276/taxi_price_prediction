import joblib
import pandas as pd
import numpy as np
from scripts.data_preprocessing import preprocess_data


def load_model(model_path='./artifacts/rf.pkl'):
    # Load the trained model from the artifacts folder
    model = joblib.load(model_path)
    return model


def predict_new_data(model, new_input, pipeline):
    # Transform the new input using the same pipeline used for training
    transformed_input = pipeline.transform(new_input)

    # Make predictions
    predictions = model.predict(transformed_input)
    return predictions
