from types import SimpleNamespace
import numpy as np
import tensorflow as tf
import os
from matplotlib.pyplot import imread
image_folder_path = "../data/images/"

def preprocess_labels(valueRange: list):
    with open("../data/IAUSD-lables.txt") as file_in:
        lines = []

        for line in file_in:
            split_line = line.split()
            split_line.remove(split_line[0])
            lines.append(np.array(split_line))

        final_lines = []
        for x in valueRange:
            final_lines.append(lines[x-1])

        final_return = np.vstack(final_lines)
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

def get_specific_class(wantedIndex: int):
    with open("../data/IAUSD-lables.txt") as file_in:
        lines = []

        for line in file_in:
            split_line = line.split()
            split_line.remove(split_line[0])
            lines.append(split_line)

    final_lines = []
    final_index = []
    for line in lines:
        if line[wantedIndex] == "1":
            final_lines.append(np.array(line))
            final_index.append(lines.index(line))

    final_return = np.vstack(final_lines)
        
    
    images = []
    for f in final_index:
        imagePath = image_folder_path + str(f + 1)+ ".jpg"
        image = imread(imagePath)
        image = image.astype('float32') / 255.0
        image = tf.image.resize(image, [224,224])
        images.append(image)

    final_return2 = np.array(images)
        
    return final_return, final_return2