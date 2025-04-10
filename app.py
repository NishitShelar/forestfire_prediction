from flask import Flask, request, jsonify
from flask_cors import CORS  # ← Add this
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

app = Flask(__name__)
CORS(app)  # ← Enable CORS for all routes

# 🔁 Load the trained model
with open("multinomial_logit_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔁 Label mapping
label_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

# 🧠 Preprocessing function
def preprocess_dataframe(df):
    df['temp_humid'] = df['air_temp'] * df['humidity']
    df['wind_frp'] = df['wind_speed'] * df['frp']
    df['heat_intensity'] = df['bright_ti4'] + df['bright_ti5'] + df['frp']
    df['dry_index'] = df['air_temp'] / (df['humidity'] + 1)
    df['solar_precip'] = df['solar_radiation'] / (df['precipitation'] + 1)
    df['interaction_1'] = df['air_temp'] * df['solar_radiation']
    df['lat_abs'] = np.abs(df['lat'])

    features = [
        "bright_ti4", "bright_ti5", "scan", "track", "frp",
        "precipitation", "air_temp", "humidity", "wind_speed",
        "solar_radiation", "type", "lat", "lon",
        "temp_humid", "wind_frp", "heat_intensity", "dry_index",
        "solar_precip", "interaction_1", "lat_abs"
    ]

    df['type'] = df['type'].astype(int)
    df = df[features]
    df_const = sm.add_constant(df)

    return df_const

# 📩 Upload and Predict Route
@app.route('/predict_csv', methods=['POST'])
def predict_from_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    try:
        df = pd.read_csv(file)

        processed_df = preprocess_dataframe(df)
        predictions = model.predict(processed_df)
        predicted_classes = predictions.idxmax(axis=1)
        predicted_labels = predicted_classes.map(label_mapping)

        df['predicted_zone'] = predicted_labels
        result_df = df[['lat', 'lon', 'predicted_zone']]
        return result_df.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🚀 Run server
if __name__ == '__main__':
    app.run(debug=True)
