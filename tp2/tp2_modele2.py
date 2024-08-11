import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pathlib
import matplotlib.pyplot as plt

# Load your data
data_dir = pathlib.Path(__file__).parent / 'TP2-images'
batch_size = 1100
img_height = 80
img_width = 80

class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])

# Load the dataset and obtain class names
train_ds = tf.keras.utils.image_dataset_from_directory(
 data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
 validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)
        print(f"Epoch {epoch+1}: Learning Rate is {lr}")


# Retrieve class names directly from the `image_dataset_from_directory` call
num_classes = len(class_names)

# Define the model
model = tf.keras.Sequential([
tf.keras.layers.Rescaling(1./255),
tf.keras.layers.Conv2D(32, 3, activation='relu'),
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(num_classes)
])

# Compile the model
model.compile(
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer='adam',
metrics=['accuracy']
)


lr_logger = LearningRateLogger()

# Train the model
history = model.fit(
train_ds,
 validation_data=val_ds,
epochs=200,
callbacks=[lr_logger]
)

lr_logger = LearningRateLogger()

num_samples_train = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Nombre total d'échantillons d'entraînement : {num_samples_train}")



# Plotting function for accuracy and loss
def plot_metrics(history, lr_logger, filename='training_validation_metrics_with_lr.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    learning_rates = lr_logger.learning_rates
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Learning Rate plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, learning_rates, label='Learning Rate')
    plt.legend(loc='upper right')
    plt.title('Learning Rate')

    # Save the plots to a PNG file
    plt.savefig(filename)
    plt.close()

# Example usage with history from model training
# Correct the function call by passing `filename` as a keyword argument
plot_metrics(history, lr_logger=lr_logger, filename='training_validation_metrics_with_lr.png')

loss, val_accuracy = model.evaluate(val_ds)
print(f"Précision de validation globale : {val_accuracy:.4f}")



# Evaluate the model
def evaluate_model(validation_data):
    y_true = []
    y_pred = []
    
    for images, labels in validation_data:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')  # Save confusion matrix plot as an image file
    plt.close()

evaluate_model(val_ds)

model.summary()
