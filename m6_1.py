import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializations
from keras.layers import merge, Dropout
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import Model



class M6_1(object):
    def __init__(self, batch_size=32, row=64, col=64, num_classes=None):
        self.batch_size = batch_size
        self.row = row
        self.col = col
        self.num_classes = num_classes
    def build_model(self):
        def my_init(shape, name=None):
            return initializations.normal(shape, scale=0.1, name=name)
        input_shape = (self.row, self.col, 1)
        img_input = Input(shape=input_shape)

        x = Convolution2D(32, 3, 3, input_shape=input_shape,
                          border_mode='same', init=my_init)(img_input)
        x = Activation('relu')(x)
        x = Convolution2D(32, 3, 3, init=my_init, border_mode='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.5)(x)

        x = Convolution2D(64, 3, 3, border_mode='same', init=my_init)(x)
        x = Activation('relu')(x)
        x = Convolution2D(64, 3, 3, border_mode='same', init=my_init)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(256, init=my_init)(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes)(x)

        x = Activation('softmax')(x)

        model = Model(img_input, x)
        return model
