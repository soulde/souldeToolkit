import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf


class InceptionV1(keras.Model):
    def __init__(self, output_channels1, output_channels2, output_channels3, output_channels4):
        super(InceptionV1, self).__init__(name="Inception v1")

        # path1: 1*1 Conv layers
        self.p1_1 = layers.Conv2D(output_channels1, kernel_size=(1, 1))

        # path2: 1*1 Conv layers and 3*3 Conv layers
        self.p2_1 = layers.Conv2D(output_channels2[0], kernel_size=(1, 1))
        self.p2_2 = layers.Conv2D(output_channels2[1], kernel_size=(3, 3), padding='same')

        # path3: 1*1 Conv layers and 5*5 Conv layers
        self.p3_1 = layers.Conv2D(output_channels3[0], kernel_size=(1, 1))
        self.p3_2 = layers.Conv2D(output_channels3[1], kernel_size=(5, 5), padding='same')

        # path4: 3*3 MaxPooling and 1*1 Conv layers
        self.p4_1 = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')
        self.p4_2 = layers.Conv2D(output_channels4, kernel_size=(1, 1))

    def call(self, inputs):
        p1 = keras.activations.relu(self.p1_1(inputs))
        p2 = keras.activations.relu(self.p2_2(keras.activations.relu(self.p2_1(inputs))))
        p3 = keras.activations.relu(self.p3_2(keras.activations.relu(self.p3_1(inputs))))
        p4 = keras.activations.relu(self.p4_2(self.p4_1(inputs)))

        return tf.concat([p1, p2, p3, p4], axis=3)


class InceptionV2(keras.Model):
    def __init__(self, output_channels1, output_channels2, output_channels3, output_channels4):
        super(InceptionV2, self).__init__(name="Inception v2")

        # path1: 1*1 Conv layers
        self.p1_1 = layers.Conv2D(output_channels1, kernel_size=(1, 1))

        # path2: 1*1 Conv layers and 3*3 Conv layers
        self.p2_1 = layers.Conv2D(output_channels2[0], kernel_size=(1, 1))
        self.p2_2 = layers.Conv2D(output_channels2[1], kernel_size=(3, 3), padding='same')

        # path3: 1*1 Conv layers and 5*5 Conv layers
        self.p3_1 = layers.Conv2D(output_channels3[0], kernel_size=(1, 1))
        self.p3_2 = layers.Conv2D(output_channels3[1], kernel_size=(3, 3), padding='same')
        self.p3_3 = layers.Conv2D(output_channels3[2], kernel_size=(3, 3), padding='same')

        # path4: 3*3 MaxPooling and 1*1 Conv layers
        self.p4_1 = layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')
        self.p4_2 = layers.Conv2D(output_channels4, kernel_size=(1, 1))

    def call(self, inputs):
        p1 = keras.activations.relu(self.p1_1(inputs))
        p2 = keras.activations.relu(self.p2_2(keras.activations.relu(self.p2_1(inputs))))
        p3 = keras.activations.relu(
            self.p3_3(keras.activations.relu(self.p3_2(keras.activations.relu(self.p3_1(inputs))))))
        p4 = keras.activations.relu(self.p4_2(self.p4_1(inputs)))

        return tf.concat([p1, p2, p3, p4], axis=3)
