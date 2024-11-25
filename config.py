from pydantic import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "model.pth")  # Default to 'model.pth' if not set in .env
    image_size: int = int(os.getenv("IMAGE_SIZE", 224))  # Default to 224
    normalization_mean: List[float] = list(map(float, os.getenv("NORMALIZATION_MEAN", "[0.485, 0.456, 0.406]").strip('[]').split(',')))
    normalization_std: List[float] = list(map(float, os.getenv("NORMALIZATION_STD", "[0.229, 0.224, 0.225]").strip('[]').split(',')))

    class Config:
        env_file = ".env"  # This tells Pydantic to load from the .env file

# Create config object
config = Config()
