from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100, max_depth=5, min_samples_leaf=2)
    rf.fit(X_train, y_train)

    # Save the trained model in the artifacts folder
    joblib.dump(rf, './artifacts/rf.pkl')

    return rf
