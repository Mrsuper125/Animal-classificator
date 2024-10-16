from argparse import ArgumentError
import numpy as np

classes = {
    0: "Заяц", 1: "Кабан", 2: "Кошки", 3: "Куньи", 4: "Медведь", 5: "Оленевые", 6: "Пантеры", 7: "Полорогие",
    8: "Собачие", 9: 'Сурок'
}


def dump_to_cvs(file_names, predictions):
    if len(file_names) != len(predictions):
        raise ArgumentError(argument=None, message="predictions length does not match file names length")
    with open("result/result.csv", "w") as file:
        file.write("image_name,predicted_class\n")
        for i in range(len(file_names)):
            file.write(f"{file_names[i]},{np.argmax(predictions[i])}\n")
