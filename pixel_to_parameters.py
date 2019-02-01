from roipoly import RoiPoly
import matplotlib.image as Image
import cv2
import sys
import numpy as np
import os
from scipy.stats import multivariate_normal as mvn



def EM(data, numOfClass, loops, savePath):
    dimension = data.shape[1]
    n = data.shape[0]
    mean = data[np.random.choice(n, numOfClass, False), :]
    covariance =  [np.eye(dimension)] * numOfClass
    p = np.zeros(shape = (n, numOfClass))
    for i in range(numOfClass):
        covariance[i] = 100 * np.multiply(covariance[i], np.random.rand(dimension, dimension))
    pre = np.array([1.0/numOfClass] * numOfClass)

    for i in range(loops):
        for c in range(numOfClass):
            currMean = mean[c]
            currCovariance = covariance[c]
            p[:,c] = (pre[c] * mvn.pdf(data, mean=currMean, cov=currCovariance, allow_singular=True)).reshape((n,))
        p = (p.T / np.sum(p, axis=1)).T
        cp = np.sum(p, axis = 0)
        for c in range(numOfClass):
            mean[c] = np.average(data, axis=0, weights=p[:,c])
            x_mean = np.matrix(data - mean[c])
            pre[c] = cp[c] * 1.0 / n
            covariance[c] = np.array(1 / cp[c] * np.dot(np.multiply(x_mean.T, p[:, c]), x_mean))
    np.save(savePath + "_pre", pre)
    np.save(savePath + "_cov", covariance)
    np.save(savePath + "_mean", mean)

def singleGaussian(data, savePath):
    mean = np.array([np.mean(data, axis = 0)])
    cov = np.array([np.cov(data, rowvar = False)])
    pre = np.array([[1]])
    np.save(savePath + "_pre", pre)
    np.save(savePath + "_cov", cov)
    np.save(savePath + "_mean", mean)

if __name__ == '__main__':

    PICTURE_PATH = "./trainset/origin_picture/"
    LABEL_PATH = "./trainset/label_set/"
    SUFFIX = ".png"
    PIX_PATH = "./trainset/sample_pixel_set/"
    PARAMETER_PATH = "./trainset/parameter/"

    data = np.load(PIX_PATH + "true_foreground_pixel.npy")
    num_of_true_foreground_pixel = data.shape[0]
    EM(data, 8, 200, PARAMETER_PATH + "true_foreground")
    # singleGaussian(data, PARAMETER_PATH + "true_foreground")

    data = np.load(PIX_PATH + "false_foreground_pixel.npy")
    num_of_false_foreground_pixel = data.shape[0]
    EM(data, 8, 200, PARAMETER_PATH + "false_foreground")
    # singleGaussian(data, PARAMETER_PATH + "false_foreground")

    data = np.load(PIX_PATH + "background_pixel.npy")
    num_of_background_pixel = data.shape[0]
    # EM(data[0:800000], PARAMETER_PATH + "background")
    singleGaussian(data, PARAMETER_PATH + "background")

    sum = num_of_true_foreground_pixel + num_of_false_foreground_pixel + num_of_background_pixel
    priorProbability = np.array([(num_of_true_foreground_pixel + 0.0) / sum,
                                 (num_of_false_foreground_pixel + 0.0) / sum,
                                 (num_of_background_pixel + 0.0) / sum
                                 ], dtype = np.float64)
    np.save(PARAMETER_PATH + "priorProbability_T_F_G", priorProbability)