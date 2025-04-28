from flask import Flask, request, jsonify
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Create Flask app
app = Flask(__name__)

# Download and load model
model_path = hf_hub_download(
    repo_id="Novadotgg/Crop-recommendation",
    filename="crop.pkl",
)
model = joblib.load(model_path)

# Feature names
FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.route('/')
def home():
    return "Crop Recommendation API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.json
        values = [data.get(name) for name in FEATURE_NAMES]

        # Check if any value is missing
        if None in values:
            return jsonify({'error': 'Missing one or more feature values'}), 400

        # Prepare the feature DataFrame
        features = pd.DataFrame([values], columns=FEATURE_NAMES)

        # Get prediction
        prediction = model.predict(features)

        # Return the predicted crop
        return jsonify({'recommended_crop': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
