# Written in tf version 1.x and keras ver 2.3

import tensorflow as tf
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Input
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow import tensordot

def ACM(x, blockname, channels=1024, groups=32):
    b, w, h, c = K.int_shape(x)
    mu = tf.reduce_mean(x, axis=[1,2], name=blockname+'_mu')
    mu = tf.expand_dims(mu, axis=1)
    mu = tf.expand_dims(mu, axis=1)
    P = Conv2D(channels//2, 1, padding='same', groups=groups, name=blockname+'_P1')(mu)
    P = relu(P)
    P = Conv2D(channels, 1, padding = 'same', groups=groups, name=blockname+'_P2')(P)
    P = sigmoid(P)

    x_mu = x - mu
    k = Conv2D(channels, 1, padding='same', groups=groups, name=blockname+'_K')(x_mu)
    q = Conv2D(channels, 1, padding='same', groups=groups, name=blockname+'_Q')(x_mu)
    k = softmax(k)
    q = softmax(q)
    k = x_mu * k
    q = x_mu * q
    k = K.sum(k, axis=[1,2])
    q = K.sum(q, axis=[1,2])
    k_q = k-q
    y = x + k_q
    y = y * P

    return y


if __name__ == '__main__':
    input_tensor = Input(shape=(256,256,1024))
    acm = ACM(input_tensor, blockname='block1')
    model = Model(inputs=input_tensor, outputs=[acm])
    model.summary()
