import numpy as np
from flask import Flask, request, render_template
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('Breast_Cancer_final.joblib')

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission and model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all the input features from the form
        features_list = [
            'area_mean', 'smoothness_mean', 'concavity_mean', 'concave_points_mean', 'fractal_dimension_mean',
            'perimeter_se', 'area_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'radius_worst',
            'texture_worst', 'perimeter_worst', 'area_worst', 'concavity_worst'
        ]
        
        # Convert the features into a list of floats
        int_features = [float(request.form.get(feature)) for feature in features_list]
        
        # Convert the features into a 2D numpy array (model expects this format)
        features = [np.array(int_features)]
        
        # Make the prediction using the loaded model
        prediction = model['model'].predict(features)
        
        # Convert the prediction to human-readable format
        output = 'Malignant' if prediction[0] == 'M' else 'Benign'

        # Render the form again with the prediction
        return render_template('index.html', prediction=output)
    
    except Exception as e:
        # Handle errors (like input parsing issues)
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
