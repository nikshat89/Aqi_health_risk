import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AI Air Quality & Environmental Health Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-setup and train if artifacts don't exist
if not os.path.exists("models/aqi_model.pkl") or not os.path.exists(
    "data/processed_aqi.csv"
):
  with st.spinner(
      "First-time setup: Processing AQI dataset and training ML models..."
  ):
    from data.process_data import process_dataset
    from train_model import train

    process_dataset()
    train()


@st.cache_resource
def load_artifacts():
  model = joblib.load("models/aqi_model.pkl")
  scaler = joblib.load("models/scaler.pkl")
  features = joblib.load("models/features.pkl")
  df = pd.read_csv("data/processed_aqi.csv")
  return model, scaler, features, df


model, scaler, features, df_stations = load_artifacts()


def assess_health_risk(aqi, age_group, condition, exposure_hours):
  base_score = (aqi / 500) * 60
  age_weight = {"Child (<12)": 12, "Adult (12-60)": 5, "Senior (60+)": 15}[
      age_group
  ]
  cond_weight = {
      "None": 0,
      "Asthma / Respiratory": 18,
      "Cardiovascular Disease": 16,
      "Pregnant": 14,
  }[condition]
  exposure_weight = min(exposure_hours * 1.5, 12)

  total_risk = min(
      round(base_score + age_weight + cond_weight + exposure_weight, 1), 100
  )

  if total_risk < 25:
    return (
        total_risk,
        "Low Risk",
        "#2ecc71",
        "Air quality is safe for normal outdoor activities.",
    )
  elif total_risk < 50:
    return (
        total_risk,
        "Moderate Risk",
        "#f1c40f",
        "Sensitive individuals should limit prolonged outdoor exertion.",
    )
  elif total_risk < 75:
    return (
        total_risk,
        "High Risk",
        "#e67e22",
        "Wear an N95 mask outdoors; keep indoor air purifiers running.",
    )
  else:
    return (
        total_risk,
        "Severe / Critical Risk",
        "#e74c3c",
        (
            "Avoid all outdoor physical activity. High probability of acute"
            " respiratory/cardiac distress."
        ),
    )


# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Go to",
    [
        "🚀 Live Prediction & Health Risk Engine",
        "📈 Model Performance & Evaluation",
        "📊 Exploratory Data Analysis (EDA)",
        "🗺️ Geospatial & Regional Analytics",
    ],
)

if app_mode == "🚀 Live Prediction & Health Risk Engine":
  st.title(
      "🌍 AI-Based Air Quality Prediction & Environmental Health Risk Engine"
  )

  with st.sidebar:
    st.markdown("---")
    st.header("1. Station & Sensor Selection")
    mode = st.radio(
        "Input Mode", ["Select Real Station from Dataset", "Custom Manual Sliders"]
    )

    if mode == "Select Real Station from Dataset":
      selected_state = st.selectbox(
          "Select State", sorted(df_stations["state"].unique())
      )
      city_options = sorted(
          df_stations[df_stations["state"] == selected_state]["city"].unique()
      )
      selected_city = st.selectbox("Select City", city_options)
      station_options = df_stations[df_stations["city"] == selected_city][
          "station"
      ].tolist()
      selected_station = st.selectbox("Select Station", station_options)

      row = df_stations[df_stations["station"] == selected_station].iloc[0]
      pm25 = st.slider(
          "PM2.5 (µg/m³)", 0.0, 400.0, float(row.get("PM2.5", 50.0))
      )
      pm10 = st.slider(
          "PM10 (µg/m³)", 0.0, 500.0, float(row.get("PM10", 80.0))
      )
      no2 = st.slider("NO₂ (µg/m³)", 0.0, 200.0, float(row.get("NO2", 30.0)))
      so2 = st.slider("SO₂ (µg/m³)", 0.0, 100.0, float(row.get("SO2", 15.0)))
      co = st.slider("CO (mg/m³)", 0.0, 10.0, float(row.get("CO", 1.0)))
      o3 = st.slider("O₃ (µg/m³)", 0.0, 180.0, float(row.get("OZONE", 40.0)))
      nh3 = st.slider("NH₃ (µg/m³)", 0.0, 200.0, float(row.get("NH3", 20.0)))
      temp = st.slider(
          "Temperature (°C)", -5.0, 48.0, float(row.get("Temperature", 30.0))
      )
      humidity = st.slider(
          "Humidity (%)", 10.0, 100.0, float(row.get("Humidity", 50.0))
      )
      wind = st.slider(
          "Wind Speed (m/s)", 0.1, 20.0, float(row.get("Wind_Speed", 3.0))
      )
    else:
      pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 400.0, 75.0)
      pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 120.0)
      no2 = st.slider("NO₂ (µg/m³)", 0.0, 200.0, 40.0)
      so2 = st.slider("SO₂ (µg/m³)", 0.0, 100.0, 15.0)
      co = st.slider("CO (mg/m³)", 0.0, 10.0, 1.1)
      o3 = st.slider("O₃ (µg/m³)", 0.0, 180.0, 45.0)
      nh3 = st.slider("NH₃ (µg/m³)", 0.0, 200.0, 25.0)
      temp = st.slider("Temperature (°C)", -5.0, 48.0, 28.0)
      humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
      wind = st.slider("Wind Speed (m/s)", 0.1, 20.0, 3.5)

    st.markdown("---")
    st.header("2. Personal Health Profile")
    age_group = st.selectbox(
        "Age Group", ["Adult (12-60)", "Child (<12)", "Senior (60+)"]
    )
    condition = st.selectbox(
        "Pre-existing Condition",
        [
            "None",
            "Asthma / Respiratory",
            "Cardiovascular Disease",
            "Pregnant",
        ],
    )
    exposure = st.slider("Daily Outdoor Exposure (Hours)", 0.0, 12.0, 2.0)

  # Inference
  input_data = {
      "PM2.5": pm25,
      "PM10": pm10,
      "NO2": no2,
      "SO2": so2,
      "CO": co,
      "OZONE": o3,
      "NH3": nh3,
      "Temperature": temp,
      "Humidity": humidity,
      "Wind_Speed": wind,
  }
  input_df = pd.DataFrame([input_data])[features]
  input_scaled = scaler.transform(input_df)
  predicted_aqi = round(float(model.predict(input_scaled)[0]), 1)

  risk_score, risk_label, risk_color, advisory = assess_health_risk(
      predicted_aqi, age_group, condition, exposure
  )

  # Top Metrics
  col1, col2, col3 = st.columns(3)
  col1.metric("Predicted AQI", f"{predicted_aqi}")
  col2.metric("Personal Health Risk Index", f"{risk_score} / 100")
  col3.markdown(
      f"### Risk Category: <span style='color:{risk_color}'>{risk_label}</span>",
      unsafe_allow_html=True,
  )

  st.info(f"**Health Advisory:** {advisory}")

  # Visualizations
  left_col, right_col = st.columns(2)

  with left_col:
    st.subheader("Health Risk Gauge")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 25], "color": "#eafaf1"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 75], "color": "#fdf2e9"},
                    {"range": [75, 100], "color": "#fdedec"},
                ],
            },
        )
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

  with right_col:
    st.subheader("Explainable AI (SHAP Pollutant Impact)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)[0]
    shap_df = pd.DataFrame(
        {"Feature": features, "Impact": shap_values}
    ).sort_values(by="Impact", key=abs, ascending=True)

    fig_shap = go.Figure(
        go.Bar(
            x=shap_df["Impact"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=[
                "#e74c3c" if val > 0 else "#3498db" for val in shap_df["Impact"]
            ],
        )
    )
    fig_shap.update_layout(
        xaxis_title="Contribution to AQI Prediction", yaxis_title="Parameter"
    )
    st.plotly_chart(fig_shap, use_container_width=True)

elif app_mode == "📈 Model Performance & Evaluation":
  st.title("📈 Machine Learning Model Performance & Evaluation")
  st.markdown(
      "Compare accuracy metrics ($R^2$ Score, RMSE, MAE) across baseline and"
      " advanced ensemble regression models trained on national station"
      " data."
  )

  # Train models dynamically for display metrics
  X = df_stations[features]
  y = df_stations["AQI"]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  sc = StandardScaler()
  X_tr_sc = sc.fit_transform(X_train)
  X_te_sc = sc.transform(X_test)

  eval_models = {
      "Ridge Regression (Baseline)": Ridge(alpha=1.0),
      "Random Forest Regressor": RandomForestRegressor(
          n_estimators=150, random_state=42
      ),
      "Gradient Boosting Regressor": GradientBoostingRegressor(
          n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
      ),
  }

  perf_data = []
  predictions_dict = {}

  # FIX: Indentation corrected here so all models run inside the loop
  for m_name, m_obj in eval_models.items():
    m_obj.fit(X_tr_sc, y_train)
    preds = m_obj.predict(X_te_sc)
    predictions_dict[m_name] = preds
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    perf_data.append({
        "Model": m_name,
        "R² Score": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
    })

  perf_df = pd.DataFrame(perf_data)

  st.subheader("📊 Performance Metrics Comparison Table")
  st.dataframe(perf_df, use_container_width=True)

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Actual vs. Predicted AQI (Gradient Boosting)")
    gb_preds = predictions_dict["Gradient Boosting Regressor"]
    fig_scatter_eval = px.scatter(
        x=y_test,
        y=gb_preds,
        labels={"x": "Observed AQI", "y": "Predicted AQI"},
        title="Observed vs Model Predicted AQI",
    )
    min_val, max_val = min(y_test.min(), gb_preds.min()), max(
        y_test.max(), gb_preds.max()
    )
    fig_scatter_eval.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="red", dash="dash"),
    )
    st.plotly_chart(fig_scatter_eval, use_container_width=True)

  with col2:
    st.subheader("Feature Importance (Gradient Boosting)")
    gb_model = eval_models["Gradient Boosting Regressor"]
    imp_df = pd.DataFrame(
        {"Feature": features, "Importance": gb_model.feature_importances_}
    ).sort_values(by="Importance", ascending=True)
    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Relative Gini Importance of Pollutants",
        color="Importance",
        color_continuous_scale="Greens",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

elif app_mode == "📊 Exploratory Data Analysis (EDA)":
  st.title("📊 Exploratory Data Analysis & Feature Correlations")
  pollutant_box_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
  selected_pollutant = st.selectbox(
      "Choose Pollutant for Distribution", pollutant_box_cols
  )
  fig_box = px.box(
      df_stations,
      y=selected_pollutant,
      points="all",
      color_discrete_sequence=["#2ca02c"],
      title=f"Distribution of {selected_pollutant} across Stations",
  )
  st.plotly_chart(fig_box, use_container_width=True)

elif app_mode == "🗺️ Geospatial & Regional Analytics":
  st.title("🗺️ Geospatial Air Quality & Regional Rankings")
  state_avg = (
      df_stations.groupby("state")["AQI"]
      .mean()
      .reset_index()
      .sort_values(by="AQI", ascending=False)
  )
  fig_bar = px.bar(
      state_avg,
      x="state",
      y="AQI",
      color="AQI",
      color_continuous_scale="Reds",
      title="Average Air Quality Index (AQI) by State",
  )
  fig_bar.update_layout(xaxis_tickangle=-45)
  st.plotly_chart(fig_bar, use_container_width=True)