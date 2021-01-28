from nn.tf_layers.Inception import *
from nn.tf_tools import *


class GoogLeNetV1(keras.Model):
    def __init__(self, num_class):
        super(GoogLeNetV1, self).__init__()

        self.b1 = keras.models.Sequential([
            layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.b2 = keras.models.Sequential([
            layers.Conv2D(64, kernel_size=(1, 1)),
            layers.Conv2D(192, kernel_size=(3, 3), padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.b3 = keras.models.Sequential([
            InceptionV1(64, (96, 128), (16, 32), 32),
            InceptionV1(128, (128, 192), (32, 96), 64),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.b4 = keras.models.Sequential([
            InceptionV1(192, (96, 208), (16, 48), 64),
            InceptionV1(160, (112, 224), (24, 64), 64),
            InceptionV1(128, (128, 256), (24, 64), 64),
            InceptionV1(112, (144, 288), (32, 64), 64),
            InceptionV1(256, (160, 320), (32, 128), 128),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.b5 = keras.models.Sequential([
            InceptionV1(256, (160, 320), (32, 128), 128),
            InceptionV1(384, (192, 384), (48, 128), 128),
            layers.GlobalAvgPool2D()
        ])

        self.outputs = keras.models.Sequential([
            layers.Dropout(0.4),
            layers.Dense(num_class)
        ])

    def call(self, inputs):
        inputs = self.b1(inputs)
        inputs = self.b2(inputs)
        inputs = self.b3(inputs)
        inputs = self.b4(inputs)
        inputs = self.b5(inputs)

        return self.outputs(inputs)


if __name__ == '__main__':
    set_gpu_memory()

    net = GoogLeNetV1(1000)

    print_layers_shape(net)

