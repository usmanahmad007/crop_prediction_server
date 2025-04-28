# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib

# Load the model from HuggingFace
model = joblib.load(
    hf_hub_download("Novadotgg/Crop-recommendation", "sklearn_model.joblib")
)

# Create FastAPI app
app = FastAPI()

# Define the input data format
class PredictionRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Create a prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Prepare input for the model
    input_data = [[
        request.nitrogen,
        request.phosphorus,
        request.potassium,
        request.temperature,
        request.humidity,
        request.ph,
        request.rainfall
    ]]
    
    # Predict
    prediction = model.predict(input_data)
    
    # Return prediction
    return {"prediction": prediction[0]}
