import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pathlib


import pathlib
data_dir = pathlib.Path("../TP2-images/")
class_names =['cane', 'cavallo', 'elefante', 'farfalla', 'Fursuit', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

print(tf.__version__)
print(class_names)