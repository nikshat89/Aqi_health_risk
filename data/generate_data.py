import os
import numpy as np
import pandas as pd

def generate_dataset():
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    n_samples = 5000

    # Simulate environmental & meteorological features
    data = pd.DataFrame({
        "PM2.5": np.random.gamma(shape=2.5, scale=35.0, size=n_samples),
        "PM10": np.random.gamma(shape=3.0, scale=45.0, size=n_samples),
        "NO2": np.random.normal(loc=45, scale=20, size=n_samples).clip(5, 200),
        "SO2": np.random.normal(loc=18, scale=10, size=n_samples).clip(2, 100),
        "CO": np.random.normal(loc=1.2, scale=0.6, size=n_samples).clip(0.1, 10.0),
        "O3": np.random.normal(loc=55, scale=25, size=n_samples).clip(5, 180),
        "Temperature": np.random.normal(loc=28, scale=7, size=n_samples),
        "Humidity": np.random.uniform(25, 90, size=n_samples),
        "Wind_Speed": np.random.exponential(scale=3.5, size=n_samples).clip(0.5, 20.0),
    })

    # Calculate realistic AQI based on dominant pollutants + weather dispersion
    data["AQI"] = (
        1.45 * data["PM2.5"]
        + 0.45 * data["PM10"]
        + 0.65 * data["NO2"]
        + 0.30 * data["O3"]
        + 8.5 * data["CO"]
        - 2.2 * data["Wind_Speed"]
        + np.random.normal(0, 8, n_samples)
    ).clip(15, 500)

    output_path = "data/air_quality_data.csv"
    data.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}")

if __name__ == "__main__":
    generate_dataset()
