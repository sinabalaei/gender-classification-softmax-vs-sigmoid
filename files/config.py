import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_train = os.path.join(BASE_DIR, "dataset", "Training")
path_test  = os.path.join(BASE_DIR, "dataset", "Validation")

MODEL_PATH = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_PATH, exist_ok=True)

PLOT_PATH = os.path.join(BASE_DIR, "plot", "training.png")
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

IMG_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 5
