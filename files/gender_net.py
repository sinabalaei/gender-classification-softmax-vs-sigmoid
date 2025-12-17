import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import tensorflow as tf
from tensorflow.keras import models, layers


class GenderNetBase(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")
        self.pool1 = layers.MaxPool2D(2)

        self.conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")
        self.pool2 = layers.MaxPool2D(2)

        self.conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")
        self.pool3 = layers.MaxPool2D(2)

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(100, activation="relu")
        self.out_layer = layers.Dense(2, activation="softmax")

    def call(self, x, training=False):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return self.out_layer(x)
