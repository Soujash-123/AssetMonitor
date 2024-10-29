from flask import Flask, request, jsonify
import joblib
import numpy as np
from collections import Counter
from datetime import datetime
import gc
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added import

# Suppress warnings and logging messages
warnings.filterwarnings("ignore")
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

# Initialize the Flask app
app = Flask(__name__)

# Load models
model_names = [
    'temperature_model', 
    'vibration_model', 
    'magnetic_flux_model',
    'audible_sound_model',
    'ultra_sound_model'
]
models = {name: joblib.load(f"{name}.pkl") for name in model_names}

# Define feature sets
feature_sets = {
    'temperature_model': ['temperature_one', 'temperature_two'],
    'vibration_model': ['vibration_x', 'vibration_y', 'vibration_z'],
    'magnetic_flux_model': ['magnetic_flux_x', 'magnetic_flux_y', 'magnetic_flux_z'],
    'audible_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'audible_sound'],
    'ultra_sound_model': ['vibration_x', 'vibration_y', 'vibration_z', 'ultra_sound']
}

def evaluate_machine_condition(temperature, vibration):
    if temperature < 80 and vibration < 1.8:
        return "Safe Condition"
    elif temperature < 100 and vibration < 2.8:
        return "Maintain Condition"
    else:
        return "Repair Condition"

def detect_temperature_anomaly(temperature):
    if temperature < 80:
        return "No significant temperature anomaly detected"
    elif 80 <= temperature < 100:
        return "Moderate Overheating - Check Lubrication"
    elif 100 <= temperature < 120:
        return "Significant Overheating - Possible Misalignment or Bearing Wear"
    else:
        return "Critical Overheating - Immediate Repair Needed"

def detect_vibration_anomaly(vibration):
    if vibration < 1.8:
        return "No significant vibration anomaly detected"
    elif 1.8 <= vibration < 2.8:
        return "Unbalance Fault"
    elif 2.8 <= vibration < 4.5:
        return "Misalignment Fault"
    elif 4.5 <= vibration < 7.1:
        return "Looseness Fault"
    else:
        return "Bearing Fault or Gear Mesh Fault"

def analyze_health(input_data):
    avg_temp = (float(input_data['temperature_one']) + float(input_data['temperature_two'])) / 2
    avg_vibration = (float(input_data['vibration_x']) + float(input_data['vibration_y']) + float(input_data['vibration_z'])) / 3
    condition = evaluate_machine_condition(avg_temp, avg_vibration)
    overall_health = "Healthy" if condition == "Safe Condition" else "Unhealthy"
    return {
        "machine_condition": condition,
        "temperature_analysis": detect_temperature_anomaly(avg_temp),
        "vibration_analysis": detect_vibration_anomaly(avg_vibration),
        "timestamp": datetime.now().isoformat(),
        "overall_health": overall_health
    }

def calculate_modes(input_data_array):
    mode_values = {}
    all_values = {key: [] for key in feature_sets['temperature_model'] + feature_sets['vibration_model'] + 
                  feature_sets['magnetic_flux_model'] + feature_sets['audible_sound_model'][3:]}

    for input_data in input_data_array:
        for key in all_values.keys():
            all_values[key].append(float(input_data[key]))

    for key, values in all_values.items():
        mode_values[key] = Counter(values).most_common(1)[0][0]

    return mode_values

def aggregate_predictions(predictions_list):
    aggregated = {}
    for model_name in model_names:
        key = model_name.replace('_model', '')
        model_predictions = [pred[key] for pred in predictions_list]
        if all(isinstance(x, (int, float)) for x in model_predictions):
            aggregated[key] = float(np.mean(model_predictions))
        else:
            most_common = Counter(model_predictions).most_common(1)[0][0]
            aggregated[key] = str(most_common)
    return aggregated

def predict_single_model(model_name, model, input_data):
    features = feature_sets[model_name]
    try:
        X_input = [float(input_data[feature]) for feature in features]
        prediction = model.predict([X_input])[0]
        prediction = float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction)
        return {model_name.replace('_model', ''): prediction}
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid input for {model_name}: {str(e)}"}

def predict_from_models(input_data_array):
    all_predictions = []
    
    # Threaded execution of model predictions for each input data
    with ThreadPoolExecutor() as executor:
        for input_data in input_data_array:
            futures = []
            for model_name, model in models.items():
                futures.append(executor.submit(predict_single_model, model_name, model, input_data))
                
            predictions = {}
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    return result
                predictions.update(result)
                
            all_predictions.append(predictions)

    # Aggregate predictions and modes
    aggregated_predictions = aggregate_predictions(all_predictions)
    modes = calculate_modes(input_data_array)
    
    # Health analysis based on modes
    analysis_summary = analyze_health(modes)

    result = {
        "predictions": aggregated_predictions,
        "modes": modes,
        "Analysis_summary": analysis_summary
    }

    # Run garbage collection to release unreferenced memory
    gc.collect()

    return result

@app.route('/predict', methods=['POST'])
def predict():
    input_data_array = request.get_json()
    logging.info(f"Received input data: {input_data_array}")
    
    if not isinstance(input_data_array, list):
        return jsonify({"error": "Input must be an array"}), 400
    if len(input_data_array) > 1800:
        return jsonify({"error": "Input array exceeds maximum length of 1800"}), 400
    if len(input_data_array) == 0:
        return jsonify({"error": "Input array cannot be empty"}), 400
    
    try:
        result = predict_from_models(input_data_array)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
