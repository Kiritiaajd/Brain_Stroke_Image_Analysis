# CT SCAN BRAIN STROKE IMAGE ANALYSIS
**Description:**  
Project Title: Brain Stroke Prediction Using CT Scan Images with Convolutional Neural Networks (CNN)

Description:

This project leverages PyTorch to develop and train a Convolutional Neural Network (CNN) model for the classification of brain stroke CT scan images into two categories: hemorrhagic and ischemic. Brain stroke detection is a critical task in the medical field, as timely and accurate diagnosis can significantly improve patient outcomes. The goal of this project is to build a robust and efficient model that achieves high accuracy in classifying brain stroke types from medical images.

Objective:

Primary Goal: Develop a reliable AI model that accurately identifies the type of brain stroke to assist radiologists and medical professionals in diagnosis.
Sub-Goals:
Achieve high accuracy on the training, validation, and test datasets.
Optimize the CNN architecture for better performance and generalization.
Implement techniques like data augmentation, batch normalization, and dropout to improve model robustness.
Dataset: The dataset consists of CT scan images categorized into two classes:

Hemorrhagic Stroke
Ischemic Stroke
The dataset is fully labeled and split into three subsets:

Training Set: Used to train the CNN model.
Validation Set: Used to tune hyperparameters and monitor overfitting.
Test Set: Used to evaluate the model's final performance.
The dataset includes both grayscale and RGB images in JPG format.

Methodology:

Data Preprocessing:

Resizing images to a uniform dimension for model compatibility.
Normalizing pixel values for consistent input distribution.
Augmenting data with transformations such as rotation, flipping, and contrast adjustment to improve model generalization.
Model Architecture:

Custom CNN with the following features:
Multiple convolutional layers for feature extraction.
Batch normalization and dropout layers for regularization.
Fully connected layers for classification.
The model outputs the probability of each stroke type.
Training:

Utilized PyTorch for defining and training the model.
Loss Function: CrossEntropyLoss.
Optimizer: Adam optimizer with a learning rate scheduler for dynamic adjustment during training.
Metrics: Accuracy, precision, recall, and F1-score.
Evaluation:

The model achieved the following performance metrics:
Training Accuracy: 94%
Validation Accuracy: 94%
Test Accuracy: 92%
Confusion matrices and ROC curves were generated to assess classification performance.
Results: The trained CNN model demonstrates reliable performance with high accuracy and generalization capabilities. It effectively classifies CT scan images into hemorrhagic or ischemic stroke types, offering potential for integration into medical diagnostic workflows.

Future Scope:

Expand the dataset to include more diverse cases and patient demographics.
Implement advanced architectures such as ResNet or EfficientNet to further improve performance.
Deploy the model as a web or mobile application for real-time predictions.
Explore transfer learning for better accuracy with smaller datasets.
Conclusion: This project highlights the potential of deep learning in medical imaging applications, particularly in assisting healthcare professionals with automated diagnosis. By leveraging PyTorch, the project successfully builds a robust, scalable, and efficient model for brain stroke prediction.

---

## Table of Contents
1. [Features](#features)
2. [Setup and Installation](#setup-and-installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)

---

## Features
- Built with **PyTorch 2.2.0** for state-of-the-art deep learning capabilities.
- Compatible with **TorchVision** for easy data augmentation and pretrained models.
- Designed for [specific task or goal].
- Modular and scalable codebase for easy customization.

---

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- Pip or Conda for package management

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Kiritiaajd/Brain_Stroke_Image_Analysis.git
   cd BRAIN-STROKE-PREDICTION
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If you encounter dependency issues (e.g., `torch` and `torchvision`), ensure compatible versions are installed:
   ```bash
   pip install torch==2.2.0 torchvision==0.17.0
   ```

---

## Usage

### Running the Project
To start the training or evaluation process, run:
```bash
python main.py
```

### Dataset Preparation
Ensure your dataset is organized as follows:
```
data/
├── train/
├── test/
└── validation/
```

Update the dataset path in the `config.yaml` or relevant file in the repository.

### Testing the Model
To evaluate the model's performance:
```bash
python evaluate.py --model_path path/to/saved/model.pth
```

---

## Project Structure
```
project-name/
├── data/                # Dataset folder
├── models/              # PyTorch models
├── scripts/             # Helper scripts
├── notebooks/           # Jupyter notebooks for experimentation
├── main.py              # Entry point for training
├── evaluate.py          # Script for evaluation
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## Dependencies
This project relies on the following key libraries:
- PyTorch 2.2.0
- TorchVision 0.17.0
- NumPy
- Matplotlib
- Pandas
- Scikit-Learn

For a full list, see [`requirements.txt`](requirements.txt).

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---


