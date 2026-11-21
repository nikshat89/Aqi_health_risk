import os
import numpy as np
import pandas as pd


def process_dataset():
    os.makedirs("data", exist_ok=True)
    raw_path = "data/AQI.csv"

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please place AQI.csv inside the data/ folder.")
        return

    df = pd.read_csv(raw_path)

    # Pivot long format pollutants into wide station features
    pivot_df = df.pivot_table(
        index=[
            "country",
            "state",
            "city",
            "station",
            "last_update",
            "latitude",
            "longitude",
        ],
        columns="pollutant_id",
        values="pollutant_avg",
    ).reset_index()

    # Handle missing pollutant values with median imputation
    pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
    for col in pollutant_cols:
        if col in pivot_df.columns:
            pivot_df[col] = pivot_df[col].fillna(pivot_df[col].median())
        else:
            pivot_df[col] = 20.0

    # Simulate meteorological dispersion factors
    np.random.seed(42)
    n = len(pivot_df)
    pivot_df["Temperature"] = np.random.normal(loc=30, scale=6, size=n).clip(10, 48)
    pivot_df["Humidity"] = np.random.uniform(30, 85, size=n)
    pivot_df["Wind_Speed"] = np.random.exponential(scale=3.0, size=n).clip(0.5, 18.0)

    # Compute composite AQI based on weighted pollutant concentrations & weather
    pivot_df["AQI"] = (
        1.5 * pivot_df.get("PM2.5", 30)
        + 0.5 * pivot_df.get("PM10", 50)
        + 0.6 * pivot_df.get("NO2", 30)
        + 0.4 * pivot_df.get("OZONE", 40)
        + 7.0 * pivot_df.get("CO", 1.0)
        + 0.3 * pivot_df.get("SO2", 15)
        + 0.2 * pivot_df.get("NH3", 20)
        - 2.0 * pivot_df["Wind_Speed"]
        + np.random.normal(0, 5, n)
    ).clip(20, 500)

    output_path = "data/processed_aqi.csv"
    pivot_df.to_csv(output_path, index=False)
    print(f"Processed dataset saved successfully at {output_path} ({n} stations)")


if __name__ == "__main__":
    process_dataset()
