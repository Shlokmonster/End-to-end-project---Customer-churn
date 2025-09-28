import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Load the model when the app starts
model = None
try:
    model = pickle.load(open('Model.sav', 'rb'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Allowed values for categorical features
ALLOWED_VALUES = {
    'Dependents': ['Yes', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No']
}

def validate_input(data, is_json=False):
    """Validate input data"""
    if is_json:
        required_fields = [
            'Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'Contract',
            'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise BadRequest(f"Missing required fields: {', '.join(missing)}")
    
    # Convert numeric fields
    try:
        data['tenure'] = float(data['tenure'])
        data['MonthlyCharges'] = float(data['MonthlyCharges'])
        data['TotalCharges'] = float(data['TotalCharges'])
    except (ValueError, TypeError):
        raise BadRequest("Numeric fields (tenure, MonthlyCharges, TotalCharges) must be valid numbers")
    
    # Validate categorical fields
    for field, allowed in ALLOWED_VALUES.items():
        if field in data and data[field] not in allowed:
            raise BadRequest(f"Invalid value for {field}. Must be one of: {', '.join(allowed)}")
    
    return data

@app.route("/")
def home_page():
    """Render the home page with the prediction form"""
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        if request.is_json:
            data = request.get_json()
            is_json = True
        else:
            data = request.form.to_dict()
            is_json = False
        
        # Validate input
        data = validate_input(data, is_json=is_json)
        
        # Prepare features for prediction
        features = [
            data['Dependents'], data['tenure'], data['OnlineSecurity'],
            data['OnlineBackup'], data['DeviceProtection'], data['TechSupport'],
            data['Contract'], data['PaperlessBilling'],
            data['MonthlyCharges'], data['TotalCharges']
        ]
        
        df = pd.DataFrame([features], columns=[
            'Dependents', 'tenure', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling',
            'MonthlyCharges', 'TotalCharges'
        ])
        
        # Encode categorical features
        categorical_features = {f for f in df.columns if df[f].dtype == 'object'}
        encoder = LabelEncoder()
        for feature in categorical_features:
            df[feature] = encoder.fit_transform(df[feature])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        result = {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': round(float(probability), 2),
            'status': 'success'
        }
        
        if is_json:
            return jsonify(result)
        else:
            op1 = f"This Customer is likely to be {result['prediction']}!"
            op2 = f"Confidence level is {result['probability']}%"
            return render_template("home.html", op1=op1, op2=op2, **data)
            
    except BadRequest as e:
        if is_json:
            return jsonify({'error': str(e), 'status': 'error'}), 400
        return render_template("home.html", error=str(e), **request.form.to_dict())
    except Exception as e:
        error_msg = "An error occurred while processing your request"
        if is_json:
            return jsonify({'error': error_msg, 'status': 'error'}), 500
        return render_template("home.html", error=error_msg, **request.form.to_dict())

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)