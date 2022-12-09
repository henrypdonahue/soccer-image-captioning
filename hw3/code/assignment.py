from types import SimpleNamespace
import numpy as np
import tensorflow as tf
import os
from matplotlib.pyplot import imread
image_folder_path = "../data/images/"

def preprocess_labels():
    with open("../data/IAUSD-lables.txt") as file_in:
        lines = []
        for line in file_in:
            split_line = line.split()
            split_line.remove(split_line[0])
            lines.append(np.array(split_line))
        final_return = np.vstack(lines)
        return final_return


def images_in_array(valueRange: list):
    images = []

    for f in valueRange:
        imagePath = image_folder_path + str(f)+ ".jpg"
        image = imread(imagePath)
        image = image.astype('float32') / 255.0
        image = tf.image.resize(image, [224,224])
        images.append(image)

    final_return = np.array(images)
    return final_return