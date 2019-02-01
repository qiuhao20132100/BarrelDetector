from roipoly import RoiPoly
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    PICTURE_PATH = "./trainset/origin_picture/"
    LABEL_PATH = "./trainset/label_set/"
    SUFFIX = ".png"
    PIX_PATH = "./trainset/sample_pixel_set/"


    true_foreground_pixel = np.array([])
    false_foreground_pixel = np.array([])
    background_pixel = np.array([])

    for index in range(1,47):
        label = plt.imread(LABEL_PATH + "Label_" + str(index) + SUFFIX)
        image = plt.imread(PICTURE_PATH + str(index) + SUFFIX)
        true_foreground_data = []
        false_foreground_data = []
        background_data = []
        for row in range(label.shape[0]):
            for col in range(label.shape[1]):
                if int(label[row][col] * 255) == 0:
                    background_data.append(image[row][col])
                elif int(label[row][col] * 255) == 1:
                    true_foreground_data.append(image[row][col])
                else:
                    false_foreground_data.append(image[row][col])
        if (len(true_foreground_data) > 0):
            if (true_foreground_pixel.size == 0) :
                true_foreground_pixel = np.array(true_foreground_data).reshape(-1,3)
            else:
                true_foreground_pixel = np.append(true_foreground_pixel, np.array(true_foreground_data).reshape(-1,3), axis = 0)

        if (len(false_foreground_data) > 0):
            if (false_foreground_pixel.size == 0) :
                false_foreground_pixel = np.array(false_foreground_data).reshape(-1,3)
            else:
                false_foreground_pixel = np.append(false_foreground_pixel, np.array(false_foreground_data).reshape(-1,3), axis = 0)

        if (len(background_data) > 0):
            if (background_pixel.size == 0) :
                background_pixel = np.array(background_data).reshape(-1,3)
            else:
                background_pixel = np.append(background_pixel, np.array(background_data).reshape(-1,3), axis = 0)

    np.save(PIX_PATH + "true_foreground_pixel", true_foreground_pixel);
    np.save(PIX_PATH + "false_foreground_pixel", false_foreground_pixel);
    np.save(PIX_PATH + "background_pixel", background_pixel);

