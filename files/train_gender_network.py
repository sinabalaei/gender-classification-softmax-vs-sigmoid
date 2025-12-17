import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from gender_net import GenderNetBase
from config import BATCH_SIZE, EPOCHS, MODEL_PATH, PLOT_PATH
from utils import generator, plot_training, TrainingMonitor, F1Score

print("[INFO] Loading dataset...")
train_gen, val_gen = generator(class_mode="categorical")

print("[INFO] Building Gender Classification Network...")
model = GenderNetBase()

print("[INFO] Compiling model...")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", F1Score()]
)

print("[INFO] Training model...")
monitor = TrainingMonitor(plot_path=PLOT_PATH)

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    callbacks=[monitor]
)

print("[INFO] Saving model...")
os.makedirs(MODEL_PATH, exist_ok=True)
model.save(os.path.join(MODEL_PATH, "gender_net.keras"))

print("[INFO] Plotting training curves...")
plot_training(history, PLOT_PATH)
