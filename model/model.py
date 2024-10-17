import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Normalization, Rescaling
from keras.api.layers import Dense, Flatten, Dropout
from keras.api.losses import SparseCategoricalCrossentropy
import keras


from keras.api.applications.resnet50 import ResNet50
from keras.api.models import Model


def make_model():
    model = ResNet50(
        weights=None,
        classes=10,
        classifier_activation="softmax", )

    """model = Sequential([
        Rescaling(1. / 255),

        Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
         MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
     MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),

        # Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
    Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
    Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
    Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'),
    MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),

        # Block 4
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'),
    MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),

        # Block 5
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
    Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'),
    MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),

    Flatten(name='flatten'),
    Dense(4096, activation='relu', name='fc1'),
    Dense(4096, activation='relu', name='fc2'),
    Dense(10),

    ])
"""
    """model = keras.applications.VGG19(
        include_top=True,
        weights=None,
        classes=10,
        classifier_activation="softmax",
        name="vgg19",
    )"""

    """
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
    """

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model
