import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import scipy.misc
from glob import glob

def generate_data(filepath, num_classes=None, row=64, col=64):
    images = read_data(filepath, num_classes)
    X_all, y_all = form_images(images, num_classes, row, col)
    return split_data(X_all, y_all, num_classes)

def read_data(filepath, num_classes):
    data = sorted(glob(filepath))
    images = []
    for image in data:
        i = scipy.misc.imread(image)
        images.append(i)
    return images


def form_images(images, num_classes, row, col):
    X_all = np.zeros((len(images), row, col))
    y_all = []
    idx = 0
    for i in range(len(images)):
        if i % (len(images)/num_classes) == 0 and i != 0:
            idx += 1
        y_all.append(idx)
        image = scipy.misc.imresize(images[i], (row, col), mode='F')/255
        X_all[i] = image
    X_all = np.reshape(X_all, [len(images), row, col, 1])
    return X_all, y_all

def split_data(X_all, y_all, num_classes):
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2)
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_test = np_utils.to_categorical(Y_test, num_classes)
    return X_train, X_test, Y_train, Y_test
