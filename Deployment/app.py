from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import CNN  # Ensure model.py is in the same directory as app.py
from PIL import Image
import io
import numpy as np
import base64  # Import base64 for decoding the image string
from torchvision import transforms

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode

app = FastAPI()

# Define input data format using Pydantic
class InputData(BaseModel):
    # Input data should be a base64 encoded image string
    image: str  # Base64 encoded image or image URL

# Image preprocessing function
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust the normalization values if needed
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    # Decode the base64 string into image bytes
    image_bytes = io.BytesIO(base64.b64decode(data.image))  # Decode base64 to bytes
    
    # Preprocess the image
    input_tensor = preprocess_image(image_bytes)
    
    # Make prediction
    with torch.no_grad():  # No need to track gradients during inference
        output = model(input_tensor)  # Forward pass through the model

    # Convert output to probabilities (softmax)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the predicted class

    return {
        "prediction": predicted_class, 
        "probabilities": probabilities.tolist()  # Return probabilities as a list
    }
