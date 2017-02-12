#from m6_1 import M6_1
from m3 import M3
from keras.preprocessing.image import ImageDataGenerator
from image_reader import *
from keras.optimizers import Adadelta

class Trainer(object):
    def __init__(self, batch_size=16, num_classes=None, row=64, col=64):
        self.num_classes = num_classes
        self.row = row
        self.col = col
        self.batch_size = batch_size
        
    def train(self):
        m6 = M3(batch_size=self.batch_size, num_classes=self.num_classes)
        model = m6.build_model()
        X_train, X_test, y_train, y_test = generate_data('./data/hiragana/*.jpg',num_classes=self.num_classes)
        datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.2)
        datagen.fit(X_train)
        optimizer = Adadelta()
        model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')
        model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=400, verbose=2,
                  validation_split=0.2)

        model.evaluate(X_test, Y_test, batch_size=self.batch_size, verbose=2)


if __name__ == '__main__':
    trainer = Trainer(num_classes=70)
    trainer.train()
