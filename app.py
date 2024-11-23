from flask import Flask, request, jsonify
import torch
from model import BrainStrokeModel  # Assuming you have a model class named `BrainStrokeModel` in a file named `model.py`

app = Flask(__name__)

# Load the model
model_path = 'Notebook/best_model.pth'  # Update this path if your model is saved elsewhere
model = BrainStrokeModel()  # Initialize your model class
model.load_state_dict(torch.load(model_path))  # Load the model weights
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions on brain stroke CT scan images.
    Request should be JSON with image data (path or raw data). 
    """
    try:
        # Get the input data
        data = request.json
        # Assuming `data` has 'image_path' key
        image_path = data.get('image_path')
        
        # Load your image and preprocess it
        # You should implement `preprocess_image` as needed for your model
        # image = preprocess_image(image_path)
        
        # For now, let's use a random tensor to simulate a prediction
        input_tensor = torch.randn(1, 3, 224, 224)  # Replace with actual preprocessing code
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert output to a suitable format
        prediction = torch.sigmoid(output).item()  # Assuming binary classification
        
        return jsonify({
            'prediction': prediction,
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
