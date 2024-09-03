from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import statsmodels.api as sm

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

label_list = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy array to list
    if isinstance(data, np.generic):
        return data.item()  # Convert NumPy scalar to Python native type
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data

app = Flask(__name__)

# Load models
log_model = joblib.load('LOG.pkl')
mle_model = joblib.load('MLE.pkl')
rf_model = joblib.load('RF.pkl')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from JSON
        data = request.get_json()
        input_data = np.array([int(data['N']), int(data['P']), int(data['K']), int(data['temperature']), int(data['humidity']), int(data['ph']), int(data['rainfall'])]).reshape(1, -1)
        input_data = sm.add_constant(input_data, has_constant='add')
        # Make predictions with all models
        log_pred = log_model.predict(input_data)[0]
        mle_pred = mle_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]

        predictions = {
            "Logistic Regression": label_list[log_pred],
            "Logistic Regression (MLE)":  label_list[mle_pred],
            "Random Forest": label_list[rf_pred]
        }
        
        # Mock prediction logic to show an example result
        best_model = max(predictions, key=lambda k: predictions[k])  

        result_data = {
            "best_model": best_model,
            "predictions": convert_to_serializable(predictions)
        }

        return jsonify(result_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 8000)