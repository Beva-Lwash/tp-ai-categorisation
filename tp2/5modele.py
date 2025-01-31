#
#
# 1er modèle  
#
#

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
img_height = 64
img_width = 64

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
metrics=['accuracy']
)


lr_logger = LearningRateLogger()

# Train the model
history = model.fit(
train_ds,
 validation_data=val_ds,
epochs=5,
callbacks=[lr_logger]
)

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

#
#
#
#2e modele
#
#
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
metrics=['accuracy']
)


lr_logger = LearningRateLogger()

# Train the model
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=300,
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
#
#
#3e modele 
#
#
#

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ' '

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pathlib

# Parameters
img_height = 80
img_width = 80
batch_size = 600

data_dir = pathlib.Path('TP2-images')

class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
 data_dir,
 validation_split=0.2,
 subset="validation",
 seed=123,
 image_size=(img_height, img_width),
 batch_size=batch_size
)

# Normalization layer
normalisation = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255)
])

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)
        print(f"Epoch {epoch+1}: Learning Rate is {lr}")

# Define data augmentation layers
data_augmentation_fursuit = tf.keras.Sequential([
 tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
tf.keras.layers.RandomZoom(0.1),
])

data_augmentation_other = tf.keras.Sequential([
 tf.keras.layers.RandomFlip("horizontal"),
 tf.keras.layers.RandomRotation(0.1),
])

# Define normalization layer
normalisation = tf.keras.layers.Rescaling(1./255)

def preprocess_image(image, label):
  class_index = tf.argmax(label)
 fursuit_class_index = class_names.index('Fursuit')
  other_classes_indices = [
  class_names.index('elefante'),
   class_names.index('gatto'),
  class_names.index('mucca'),
 class_names.index('pecora')
  ]
 
def apply_augmentation(image, augmentation_layer):
   return augmentation_layer(image)

def apply_augmentations(image):
 is_fursuit = tf.equal(class_index, fursuit_class_index)
 is_other_class = tf.reduce_any(tf.equal(class_index, other_classes_indices))
 
image = tf.cond(
  is_fursuit,
  true_fn=lambda: apply_augmentation(image, data_augmentation_fursuit),
     false_fn=lambda: tf.cond(
       is_other_class,
       true_fn=lambda: apply_augmentation(image, data_augmentation_other),
        false_fn=lambda: image
        )
      )
      return image

image = apply_augmentations(image)
return image, label


train_ds = tf.keras.utils.image_dataset_from_directory(
 data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
   data_dir,
    validation_split=0.2,
     subset="validation",
     seed=123,
     image_size=(img_height, img_width),
     batch_size=batch_size
)

# Apply preprocessing
train_ds = train_ds.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (normalisation(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation(x), y))

# Define model
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(img_height, img_width, 3)),
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy'])

# Train the model
history = model.fit(
train_ds,
validation_data=val_ds,
epochs=100,
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
#
#
#4e modele 
#
#
#

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


import pathlib

# Parameters
img_height = 80
img_width = 80
batch_size = 1000

data_dir = pathlib.Path('TP2-images')

class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])



# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
     data_dir,
     validation_split=0.2,
     subset="training",
     seed=123,
     image_size=(img_height, img_width),
     batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)





# Normalization layer
normalisation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

# Define data augmentation layers
data_augmentation_fursuit = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomZoom(0.7),
    tf.keras.layers.RandomBrightness(0.4),
    tf.keras.layers.RandomContrast(0.1),

])

data_augmentation_other = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
])

# Define normalization layer
normalisation = tf.keras.layers.Rescaling(1./255)

def preprocess_image(image, label):
    class_index = tf.argmax(label)
    # Define the classes that will use specific augmentations
    fursuit_class_index = class_names.index('Fursuit')
    other_classes_indices = [
        class_names.index('elefante'),
        class_names.index('gatto'),
        class_names.index('mucca'),
        class_names.index('pecora')
    ]
    
    def apply_augmentation(image, augmentation_layer):
        return augmentation_layer(image)

    def apply_augmentations(image):
        is_fursuit = tf.equal(class_index, fursuit_class_index)
        is_other_class = tf.reduce_any(tf.equal(class_index, other_classes_indices))
        
        image = tf.cond(
            is_fursuit,
            true_fn=lambda: apply_augmentation(image, data_augmentation_fursuit),
            false_fn=lambda: tf.cond(
                is_other_class,
                true_fn=lambda: apply_augmentation(image, data_augmentation_other),
                false_fn=lambda: image
            )
        )
        return image

    image = apply_augmentations(image)
    return image, label

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)
        print(f"Epoch {epoch+1}: Learning Rate is {lr}")



# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Apply preprocessing
train_ds = train_ds.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (normalisation(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation(x), y))




# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(units=len(class_names), activation='softmax')
])

model.compile(optimizer='adamw',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

lr_logger = LearningRateLogger()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[lr_logger]
)

model.summary()

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
#
#
#5e modele 
#
#
#

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


import pathlib

# Parameters
img_height = 80
img_width = 80
batch_size = 1000

data_dir = pathlib.Path('TP2-images')

class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])






# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
     data_dir,
     validation_split=0.2,
     subset="training",
     seed=123,
     image_size=(img_height, img_width),
     batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)





# Normalization layer
normalisation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

# Define data augmentation layers
data_augmentation_fursuit = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomZoom(0.7),
    tf.keras.layers.RandomBrightness(0.4),
    tf.keras.layers.RandomContrast(0.1),

])

data_augmentation_other = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
])

# Define normalization layer
normalisation = tf.keras.layers.Rescaling(1./255)

def preprocess_image(image, label):
    class_index = tf.argmax(label)
    # Define the classes that will use specific augmentations
    fursuit_class_index = class_names.index('Fursuit')
    other_classes_indices = [
        class_names.index('elefante'),
        class_names.index('gatto'),
        class_names.index('mucca'),
        class_names.index('pecora')
    ]
    
    def apply_augmentation(image, augmentation_layer):
        return augmentation_layer(image)

    def apply_augmentations(image):
        is_fursuit = tf.equal(class_index, fursuit_class_index)
        is_other_class = tf.reduce_any(tf.equal(class_index, other_classes_indices))
        
        image = tf.cond(
            is_fursuit,
            true_fn=lambda: apply_augmentation(image, data_augmentation_fursuit),
            false_fn=lambda: tf.cond(
                is_other_class,
                true_fn=lambda: apply_augmentation(image, data_augmentation_other),
                false_fn=lambda: image
            )
        )
        return image

    image = apply_augmentations(image)
    return image, label

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)
        print(f"Epoch {epoch+1}: Learning Rate is {lr}")



# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Apply preprocessing
train_ds = train_ds.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (normalisation(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation(x), y))

# Assuming your dataset labels are sparse (i.e., single integers per label)
labels = []
for _, label_batch in train_ds:
    labels.extend(label_batch.numpy())

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))


# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(units=len(class_names), activation='softmax')
])

model.compile(optimizer='adamw',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

lr_logger = LearningRateLogger()

history = model.fit(
    train_ds,
    class_weight=class_weights,
    validation_data=val_ds,
    epochs=100,
    callbacks=[lr_logger]
)

model.summary()

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





