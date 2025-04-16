import os
import joblib
import pickle
import pandas as pd
from scripts.model_training import train_model
from scripts.model_inference import load_model, predict_new_data
from scripts.data_preprocessing import load_and_clean_data, preprocess_data
from utils import haversine, time_cal


# Step 1: Check if the model pickle file exists
model_path = './artifacts/rf.pkl'
pipeline_path = './artifacts/pipeline5.pkl'

def run_inference():
    # Load cleaned data (make sure data preprocessing steps are applied)
    # df_clean = load_and_clean_data('data\yellow_tripdata_2015-01 .csv')

    dist = haversine(40.757336,-73.985994,40.713051,-74.007233)
    time = time_cal(origin=(40.757336,-73.985994),destination=(40.713051,-74.007233))
    new_input = pd.DataFrame([{
    'passenger_count': 3,
    'trip_distance': dist,
    'trip_duration_minutes': time
    }])

    # Load the model (if it exists)
    # print("Loading the model...")
    model = load_model(model_path)
    

    # print("Loading the preprocessing pipeline...")
    # 
    pipeline = joblib.load(pipeline_path)
    # print(pipeline)



    # Use the preprocessing pipeline (which should be available from the training process)
    transformed_input = pipeline.transform(new_input)
    # print("Transformed input:", transformed_input)

    # Make predictions
    predictions = predict_new_data(model, transformed_input, pipeline)

    print("Predictions of Cab Price :", predictions[0])


if __name__ == "__main__":
    # Step 2: If pickle model exists, use it for prediction
    if os.path.exists(model_path):
        # print("Model found! Loading the model for prediction...")
        run_inference()
    else:
        # Step 3: If pickle does not exist, train the model, save it and then predict
        print("Model not found! Training the model...")
        # Load and clean data
        df_clean = load_and_clean_data('./data/yellow_tripdata_2015-01.csv')

        # Splitting the data into features (X) and target (y)
        X = df_clean.drop(columns=['total_amount'])
        y = df_clean['total_amount']

        # Split the data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

        # Preprocess the data
        X_train_transformed, X_test_transformed, pipeline = preprocess_data(X_train, X_test)

        # Train the model
        model = train_model(X_train_transformed, y_train)

        # Save the trained model
        joblib.dump(model, model_path)
        print(f"Model trained and saved to {model_path}.")

        # After saving the model, run inference
        run_inference()
