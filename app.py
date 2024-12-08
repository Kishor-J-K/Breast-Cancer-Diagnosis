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
    return render_template('index.html')  # Corrected path

# Prediction route to handle form submission and model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        # Collect and convert form input to float values
        int_features = [float(request.form.get(feature)) for feature in model['input_cols']]
        
        # Convert the features into a 2D numpy array
        features = [np.array(int_features)]
        
        # Make a prediction using the loaded model
        prediction = model['model'].predict(features)  # Ensure your model supports this syntax
        
        # Map prediction to human-readable output
        output = 'Malignant' if prediction[0] == 'M' else 'Benign'

        # Return the prediction result to the template
        return render_template('index.html', prediction=output)
    
    except Exception as e:
        # Handle errors (e.g., input parsing issues)
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
