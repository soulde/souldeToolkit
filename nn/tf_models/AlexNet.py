from nn.tf_tools import *
from tensorflow import keras
from tensorflow.keras import layers


class AlexNet(keras.Model):
    def __init__(self, num_class):
        super(AlexNet, self).__init__()
        self.block1 = keras.models.Sequential([
            layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2),
            layers.BatchNormalization(),

            layers.Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])

        self.block2 = keras.models.Sequential([
            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),

            layers.Conv2D(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),

            layers.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])

        self.flatten = layers.Flatten()

        self.block3 = keras.models.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.25),

            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.25),

            layers.Dense(1000, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        path1 = self.block1(inputs)
        path2 = self.block1(inputs)
        feature = layers.concatenate([path1, path2], axis=3)
        print(feature.shape)

        path1 = self.block2(feature)
        path2 = self.block2(feature)
        feature = layers.concatenate([path1, path2], axis=3)
        print(feature.shape)

        return self.block3(self.flatten(feature))


if __name__ == "__main__":
    set_gpu_memory()

    net = AlexNet(1000)

    print_layers_shape(net, [227, 227, 3])
