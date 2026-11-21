import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train():
    processed_path = "data/processed_aqi.csv"
    if not os.path.exists(processed_path):
        print("Processed data not found. Running data processor...")
        from data.process_data import process_dataset
        process_dataset()

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(processed_path)
    feature_cols = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
        "Temperature",
        "Humidity",
        "Wind_Speed",
    ]
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }

    best_model = None
    best_r2 = -np.inf
    best_name = ""

    print("--- Model Evaluation on AQI Dataset ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"{name} -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    joblib.dump(best_model, "models/aqi_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/features.pkl")
    print(f"\nSaved best model ({best_name}) and preprocessing artifacts to models/")


if __name__ == "__main__":
    train()
