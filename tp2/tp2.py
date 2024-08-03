
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Charger les données
data_dir = pathlib.Path(__file__).parent / 'TP2-images'
dataset = tf.data.Dataset.list_files(str(data_dir/'*/*'))
image_count = len(list(data_dir.glob('*/*')))
batch_size = 1100
img_height = 64
img_width = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print("Classes:", class_names)

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

print("Nombre d'images:", image_count)

num_classes = len(class_names)

# Définir le modèle
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# Compiler le modèle sans spécifier d'optimiseur explicitement
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

epochs = 5 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.summary()

learning_rate = model.optimizer.learning_rate.numpy()
optimizer_name = model.optimizer.get_config()['name']

accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Learning Rate: {learning_rate}")
print(f"Optimizer: {optimizer_name}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Training Accuracy: {accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Tracer les courbes de précision et de perte
def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_metrics(history)

# Afficher l'analyse des paramètres expérimentaux
def analyse_experiment():
    print("\nAnalyse des paramètres expérimentaux:")
    print("- Learning Rate par défaut utilisé.")
    print("- Batch Size de 1100 est élevé, vérifiez la mémoire GPU.")
    print("- Epochs de 5 pour une évaluation initiale rapide.")
    print("- Validation Split de 20% est standard.")

analyse_experiment()
