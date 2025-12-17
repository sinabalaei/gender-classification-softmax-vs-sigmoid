import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths for training and validation datasets
path_train = os.path.join(BASE_DIR, "dataset", "Training")
path_test  = os.path.join(BASE_DIR, "dataset", "Validation")

# Path to save trained models
MODEL_PATH = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_PATH, exist_ok=True)  # Create folder if not exists

# Path to save training plots
PLOT_PATH = os.path.join(BASE_DIR, "plot", "training.png")
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)  # Ensure folder exists

# Image input shape (Height, Width, Channels)
IMG_SHAPE = (128, 128, 3)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
