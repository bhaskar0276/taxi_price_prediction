import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    df_clean = df.drop(columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'pickup_longitude',
                                'pickup_latitude', 'RateCodeID', 'store_and_fwd_flag', 'dropoff_longitude',
                                'dropoff_latitude', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                                'tolls_amount', 'improvement_surcharge'])
    return df_clean


def preprocess_data(X_train, X_test):
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_transformed = pd.DataFrame(pipeline.fit_transform(X_train), columns=X_train.columns)
    X_test_transformed = pd.DataFrame(pipeline.transform(X_test), columns=X_test.columns)

    return X_train_transformed, X_test_transformed, pipeline
