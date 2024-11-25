from setuptools import setup, find_packages

setup(
    name="brain-stroke-prediction",  # Replace with your package name
    version="0.1.0",  # Replace with your version number
    packages=find_packages(),  # Automatically finds all the packages
    install_requires=[
        "fastapi",  # FastAPI framework
        "uvicorn",  # ASGI server for FastAPI
        "torch",  # PyTorch for model inference
        "pydantic",  # Pydantic for input data validation
        "Pillow",  # For image processing (PIL)
        "torchvision",  # For image transforms (e.g., resizing)
        "python-dotenv",  # For loading environment variables from .env
    ],
    entry_points={
        "console_scripts": [
            "start-server = app:main",  # Optional: Add custom CLI commands
        ],
    },
)
