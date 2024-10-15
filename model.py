import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Normalization, Rescaling
from keras.api.layers import Dense, Flatten, Dropout
from keras.api.losses import SparseCategoricalCrossentropy

def make_model():
    layer = Normalization()

    layer.adapt(np.array([0.485, 0.456, 0.406],dtype='float32'))

    model = Sequential([

        Rescaling(1. / 255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model