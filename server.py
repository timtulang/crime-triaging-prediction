from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI()

# Enable CORS so your static HTML file can talk to this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the artifacts into memory on startup
print("Loading model and PSO results...")
model = joblib.load("crime_pred_pso_v1.pkl")

with open("pso_results.json", "r") as f:
    pso_data = json.load(f)

# Define the expected JSON payload from the frontend
class IncidentRequest(BaseModel):
    hour: int
    dow: int
    month: int
    loc: str
    area: int
    district: int
    domestic: int
    arrest: int

@app.get("/api/metrics")
def get_metrics():
    """Returns the real PSO results to populate the dashboard tabs."""
    return pso_data

@app.post("/api/predict")
def predict_incident(req: IncidentRequest):
    """Reconstructs the feature matrix and returns the XGBoost prediction."""
    
    # 1. Reconstruct the temporal features exactly as engineer_features() did
    hour_sin = np.sin(2 * np.pi * req.hour / 24)
    hour_cos = np.cos(2 * np.pi * req.hour / 24)
    month_sin = np.sin(2 * np.pi * req.month / 12)
    month_cos = np.cos(2 * np.pi * req.month / 12)
    dow_sin = np.sin(2 * np.pi * req.dow / 7)
    dow_cos = np.cos(2 * np.pi * req.dow / 7)
    
    is_weekend = 1 if req.dow >= 5 else 0
    is_night = 1 if req.hour >= 22 or req.hour <= 5 else 0
    quarter = (req.month - 1) // 3 + 1
    year = 2024 # Or current year
    
    # 2. Reconstruct spatial features
    HOTSPOT_AREAS = {25, 26, 27, 28, 29, 44, 67, 68, 71}
    is_hotspot = 1 if req.area in HOTSPOT_AREAS else 0
    
    # Map the frontend location string to the encoded integer from training
    # Note: You'll need to match this to how your LabelEncoder mapped them
    loc_map = {"street": 3, "residence": 2, "parking": 1, "school": 4, "bar": 5, "commercial": 6, "transit": 7}
    location_enc = loc_map.get(req.loc, 0)

    # 3. Build the single-row DataFrame. 
    # Note: For missing features like lat/lon, the model's pipeline Imputer will handle them.
    df = pd.DataFrame([{
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "is_weekend": is_weekend, "is_night": is_night,
        "quarter": quarter, "year": year,
        "latitude": np.nan, "longitude": np.nan, # Let the imputer fill these
        "dist_to_loop": np.nan, 
        "is_hotspot": is_hotspot,
        "beat": req.district * 100 + 10, # Rough proxy
        "district": req.district,
        "ward": 1, # Placeholder
        "community_area": req.area,
        "x_coordinate": np.nan, "y_coordinate": np.nan,
        "arrest": req.arrest,
        "domestic": req.domestic,
        "location_enc": location_enc
    }])

    # 4. Predict
    prob = float(model.predict_proba(df)[0][1]) # Get probability of class 1 (Violent)
    
    return {
        "probability": prob,
        "is_violent": prob >= 0.50
    }