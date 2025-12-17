import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF log messages
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import tensorflow as tf
from tensorflow.keras import layers

class GenderNetBase(tf.keras.Model):
    """
    Custom CNN model for binary gender classification.
    Designed with 3 convolutional blocks, flattening, 
    and dense layers, ending with softmax for 2-class output.
    """
    def __init__(self):
        super().__init__()

        # First conv block
        self.conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPool2D(2)

        # Second conv block
        self.conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")
        self.pool2 = layers.MaxPool2D(2)

        # Third conv block
        self.conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")
        self.pool3 = layers.MaxPool2D(2)

        # Flatten and dense layers
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(100, activation="relu")

        # Output layer: Softmax for 2-class probability distribution
        self.out_layer = layers.Dense(2, activation="softmax")

    def call(self, x, training=False):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return self.out_layer(x)
