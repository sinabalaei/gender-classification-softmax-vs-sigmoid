import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Metric
from config import path_train, path_test, IMG_SHAPE, BATCH_SIZE
import matplotlib.pyplot as plt

def datagen():

    train_datagen = ImageDataGenerator(rescale=1/255.0,
                                    rotation_range=20,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1/255.0)

    return train_datagen, test_datagen

def generator(class_mode='categorical'):

    train_datagen, test_datagen = datagen() 
      
    generator_train = train_datagen.flow_from_directory(path_train,
                                                        target_size=IMG_SHAPE[:2],
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=class_mode)
    
    generator_test = test_datagen.flow_from_directory(path_test,
                                                    target_size=IMG_SHAPE[:2],
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False, 
                                                    class_mode=class_mode)

    return generator_train, generator_test


def plot_training(history, plot_path):
    epochs = range(1, len(history.history['loss']) + 1)
    
    plt.figure(figsize=(10,6))
    
    # Loss
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history.get('val_loss', []), label='Validation Loss')
    
    # Accuracy
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history.get('val_accuracy', []), label='Validation Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(model, generator, class_mode='binary'):
    import numpy as np
    
    # پیش‌بینی
    y_true = generator.classes
    y_pred_prob = model.predict(generator)
    
    if class_mode == 'binary':
        y_pred = (y_pred_prob > 0.5).astype(int).ravel()
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=generator.class_indices)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, plot_path):
        self.plot_path = plot_path
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.best_epoch = 0
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        # ذخیره بهترین epoch
        if logs.get("val_accuracy", 0) > self.best_val_acc:
            self.best_val_acc = logs["val_accuracy"]
            self.best_epoch = epoch + 1  # index base 0

        # رسم نمودار هر epoch
        plt.figure(figsize=(8,5))
        plt.plot(self.losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.plot(self.acc, label="Train Acc")
        plt.plot(self.val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Training Progress (best epoch={self.best_epoch})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # تبدیل one-hot → class index
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
