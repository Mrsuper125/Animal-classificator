import keras.api.models

import tensorflow as tf

from utils.dataset_optimizer import configure_for_performance
from utils.dataset_retrieval import get_training_dataset, get_testing_dataset
from model.model import make_model
from utils.io import dump_to_cvs, classes

import numpy as np

train_dataset, validation_dataset = get_training_dataset("train", "train.csv")

train_dataset = configure_for_performance(train_dataset)
validation_dataset = configure_for_performance(validation_dataset)

IS_LEARNING = False

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

dump_to_cvs(file_names, predictions)
