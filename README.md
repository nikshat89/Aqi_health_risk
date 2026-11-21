# AI-Based Air Quality Prediction & Environmental Health Risk Assessment

## Included Real-World Dataset
Your uploaded `AQI.csv` (covering 448 monitoring stations across 237 cities and 30 states in India) is already included inside `data/AQI.csv`.

## Quick Start Guide

1. **Extract the ZIP file** and open your terminal inside the `aqi_health_risk_project` folder.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline & launch dashboard:**
   ```bash
   python data/process_data.py
   python train_model.py
   streamlit run app.py
   ```
   *(Note: Even if you skip steps 1 & 2 of the scripts and run `streamlit run app.py` directly, the app will automatically process `data/AQI.csv` and train the models on first startup.)*
