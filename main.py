import keras.api.models

import tensorflow as tf

from utils.dataset_optimizer import configure_for_performance
from utils.dataset_retrieval import get_training_dataset, get_testing_dataset
from model.model import make_model
from utils.io import dump_to_cvs, classes

from matplotlib import pyplot as plt

import numpy as np

train_dataset, validation_dataset = get_training_dataset("train", "train.csv")

train_dataset = configure_for_performance(train_dataset)
validation_dataset = configure_for_performance(validation_dataset)

IS_LEARNING = True

if IS_LEARNING:
    model = make_model()

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=15
    )

    model.save("saved_model/saved_model.keras")
else:
    model = keras.api.models.load_model("saved_model/model.keras")

testing_dataset, file_names = get_testing_dataset("test")

predictions = model.predict(testing_dataset)

iterated_dataset = iter(testing_dataset)

plt.figure(figsize=(10, 10))

for i in range(25):
    image_retrieved = next(iterated_dataset)
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(image_retrieved[0].numpy().astype("uint8"))
    label = np.argmax(predictions[i])
    plt.title(classes[label])
    plt.axis("off")

plt.plot()
plt.show()

dump_to_cvs(file_names, predictions)
