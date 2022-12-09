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
        #print(final_return)
        return final_return

#preprocess_labels()


def images_in_array(valueRange: list):
    images = []

    for f in valueRange:
        imagePath = image_folder_path + str(f)+ ".jpg"
        image = imread(os.path.join(imagePath, f))
        image = image.astype('float32') / 255.0
        image = tf.image.resize(image, [224,224])
        images.append(image)

    final_return = np.array(images)
    
    print(final_return.shape)
    return final_return


    # for f in os.listdir(image_folder_path):
    #     if f == '.DS_Store':
    #         continue
    #     image = imread(os.path.join(image_folder_path, f))
    #     image = image.astype('float32') / 255.0
    #     image = tf.image.resize(image, [224,224])
    #     #print(image.shape)
    #     images.append(image)

    # final_return = np.array(images)
    
    # print(final_return.shape)
    # return final_return
    
#images_in_array()


# def divide_data():
#     labels = preprocess_labels()
#     images = images_in_array()
    
#     X0 = images
#     Y0 = labels

#     return X0, Y0


# X0, Y0 = divide_data()
# print(X0.shape)
# print(Y0.shape)