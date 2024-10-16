import os
import tensorflow as tf
import keras


def load_labels(raw_labels: list[str]) -> dict[
    str, int]:  # get CSV file split into lines, return pairs of image name and group id (a.k.a. label)
    res = dict()

    for raw_label in raw_labels[1:]:
        _, group_number, image_name = raw_label.split(",")
        res[image_name[:-1]] = int(group_number)  # cut out the newline symbol and add the label to the dictionary

    return res


def apply_labels(labels: dict[str, int], directory_path: str) -> list[
    int]:  # get image-group pairs and directory name, return list of file labels in this directory
    res = []

    _, _, file_names = list(os.walk(directory_path))[0]
    for i in file_names:
        res.append(labels.get(i))  # Iterate through all files, add to the list the label of this file

    return res


def get_training_dataset(directory_path: str, labels_file: str) -> tuple[tf.data.Dataset]:  # returns
    with open(labels_file, "r") as file:
        lines = file.readlines()  # read all file-label pairs

        unsorted_labels = load_labels(lines)  # load file-label pairs into dictionary

        sorted_labels = apply_labels(unsorted_labels, directory_path)  # Get label for each file

        dataset = keras.utils.image_dataset_from_directory(directory_path, labels=sorted_labels,
                                                           label_mode="int", validation_split=0.3,
                                                           subset="both", seed=123,  # color_mode="grayscale",
                                                           image_size=(
                                                               224, 224))  # Build the dataset //TODO: image size

        return dataset

def get_testing_dataset(directory_path: str) -> (tf.data.Dataset, list[str]):
    _, _, file_names = list(os.walk(directory_path))[0]

    dataset = keras.utils.image_dataset_from_directory(directory_path, labels=None, image_size=(224, 224), shuffle=False, batch_size=1)

    return dataset, file_names
