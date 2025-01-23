from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('wine_quality_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get input data from the client
        features = np.array(data['features']).reshape(1, -1)  # Reshape the input into an array for prediction

        # Predict the wine quality using the loaded model
        prediction = model.predict(features)

        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
