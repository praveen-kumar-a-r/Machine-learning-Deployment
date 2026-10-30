from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained ML Pipeline (includes both Preprocessing and Classifier)
model_path = 'logistic_regression_model.joblib'
model_pipeline = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert input data into a Pandas DataFrame (expects keys to match dataset column names)
        input_df = pd.DataFrame([data])
        
        # Predict class
        prediction = model_pipeline.predict(input_df)[0]
        
        # Get class probabilities if supported
        try:
            probabilities = model_pipeline.predict_proba(input_df)[0]
            classes = model_pipeline.classes_
            prob_dict = {str(c): float(p) for c, p in zip(classes, probabilities)}
        except AttributeError:
            prob_dict = None

        response = {
            'status': 'success',
            'prediction': str(prediction)
        }
        if prob_dict:
            response['probabilities'] = prob_dict
            
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Logistic Regression Model API is running. Send POST requests to /predict.'
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
