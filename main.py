import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import keras
import pathlib
import matplotlib.pyplot as plt
from keras.src.losses import SparseCategoricalCrossentropy

from utils.dataset import get_dataset
from model import make_model

gpus = tf.config.list_physical_devices('GPU')

train_dataset, validation_dataset = get_dataset("train", "train.csv")

batch_size = 32

"""

image_batch, label_batch = next(iter(train_dataset))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.axis("off")
plt.plot()
plt.show()

"""
AUTOTUNE = tf.data.AUTOTUNE

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_dataset = configure_for_performance(train_dataset)
validation_dataset = configure_for_performance(validation_dataset)

model = make_model()

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15
)
