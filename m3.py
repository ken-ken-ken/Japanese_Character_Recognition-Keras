from keras.layers import Dense, Dropout
from keras.layers import Flatten, Input
from keras import backend as K
from keras.layers import Activation
from keras import initializations
from keras.models import Model

class M3(object):
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

        x = Flatten()(img_input)
        x = Dense(5000, init=my_init)(x)
        x = Dropout(0.5)(x)
        x = Dense(5000, init=my_init)(x)
        x = Dropout(0.5)(x)
        x = Dense(5000, init=my_init)(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes)(x)

        x = Activation('softmax')(x)

        model = Model(img_input, x)
        return model
