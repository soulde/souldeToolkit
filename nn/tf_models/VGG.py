from nn.tf_tools import *
from tensorflow import keras
from tensorflow.keras import layers


class VGG(keras.Model):
    def __init__(self, num_class, version='11'):
        super(VGG, self).__init__()

        assert ((version == '11') or (version == '13') or (version == '16') or (version == '19'))
        version = int(version)
        print('VGG version:', version)

        self.block1 = keras.models.Sequential([
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
        ])

        if version == 11:
            self.block1.add(layers.BatchNormalization())
        else:
            self.block1.add(layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.block1.add(layers.MaxPool2D())

        self.block2 = keras.models.Sequential([
            layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
            layers.MaxPool2D()
        ])

        if version > 11:
            self.block2.add(layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.block2.add(layers.MaxPool2D())

        self.block3 = keras.models.Sequential([
            layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
            layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
        ])

        if version > 13:
            self.block3.add(layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        if version > 16:
            self.block3.add(layers.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.block3.add(layers.MaxPool2D())

        self.block4 = keras.models.Sequential([
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
        ])

        if version > 13:
            self.block4.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        if version > 16:
            self.block4.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.block4.add(layers.MaxPool2D())

        self.block5 = keras.models.Sequential([
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
            layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
        ])

        if version > 13:
            self.block5.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        if version > 16:
            self.block5.add(layers.Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.block5.add(layers.MaxPool2D())

        self.flatten = layers.Flatten()

        self.block6 = keras.models.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(num_class, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        b1 = self.block1(inputs)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)

        return self.block6(self.flatten(b5))


if __name__ == '__main__':
    set_gpu_memory()

    net = VGG(1000, version='16')

    print_layers_shape(net, [224, 224, 3])
