from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from waitress import serve

app = Flask(__name__)

# Load the model and scaler
MODEL_PATH = 'random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def create_lag_features(data, lags):
    df = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[0].shift(lag)
    return df.dropna()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    confirmed_cases = input_data['Confirmed']

    if len(confirmed_cases) < 10:
        return jsonify({'error': 'Not enough data. Please provide at least 10 confirmed cases.'})

    df = pd.DataFrame(confirmed_cases)

    # Create lag features
    lags = 10  # Since we need 10 data points including the current point
    lagged_data = create_lag_features(df.values, lags - 1)  # lags - 1 because we want 10 columns in total

    if lagged_data.empty:
        return jsonify({'error': 'Not enough data after creating lag features. Please provide more confirmed cases.'})

    # Prepare features for prediction
    X = lagged_data.values  # This should give us the right shape
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    # Run the app with Waitress
    serve(app, host='0.0.0.0', port=5000)
