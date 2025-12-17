import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for consistent results
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # Suppress OpenCV logging

from gender_net import GenderNetBase
from config import BATCH_SIZE, EPOCHS, MODEL_PATH, PLOT_PATH
from utils import generator, plot_training, TrainingMonitor, F1Score

# Load Dataset
print("[INFO] Loading dataset...")
# Create generators for training and validation sets
train_gen, val_gen = generator(class_mode="categorical")

# Build Model
print("[INFO] Building Gender Classification Network...")
# Initialize custom CNN model
model = GenderNetBase()

# Compile Model
print("[INFO] Compiling model...")
# Using categorical_crossentropy with Softmax output
# Metrics: accuracy and custom F1-score for better evaluation
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", F1Score()]
)

# Train Model
print("[INFO] Training model...")
# Custom callback to monitor training and save plots
monitor = TrainingMonitor(plot_path=PLOT_PATH)

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    callbacks=[monitor]
)


# Save Trained Model
print("[INFO] Saving model...")
os.makedirs(MODEL_PATH, exist_ok=True)  # Ensure model folder exists
model.save(os.path.join(MODEL_PATH, "gender_net.keras"))


# Plot Training Curves
print("[INFO] Plotting training curves...")
# Save training/validation loss and accuracy curves
plot_training(history, PLOT_PATH)
