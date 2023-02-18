from nn.tf_tools import *
from tensorflow import keras
from tensorflow.keras import layers


class LeNet(keras.Model):
    def __init__(self, num_class):
        super(LeNet, self).__init__()

        self.conv1 = layers.Conv2D(32, kernel_size=(5, 5), strides=1, padding='valid', activation='relu')

        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.conv2 = layers.Conv2D(64, kernel_size=(5, 5), strides=1, padding='valid', activation='relu')

        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(128, activation='relu')

        self.fc2 = layers.Dense(num_class, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        feature = self.pool1(self.conv1(inputs))
        feature = self.pool2(self.conv2(feature))

        return self.fc2(self.fc1(self.flatten(feature)))


if __name__ == '__main__':
    set_gpu_memory()

    net = LeNet(10)

    print_layers_shape(net, [32, 32, 3])
