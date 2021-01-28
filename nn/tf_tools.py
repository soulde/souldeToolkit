import tensorflow as tf


def print_layers_shape(nn):
    x = tf.random.uniform([1, 224, 224, 3])

    for blk in nn.layers:
        x = blk(x)
        print('output shape: ', x.shape)


def set_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])