import tensorflow as tf
import numpy as np

def standardize(frame):
    return tf.image.per_image_standardization(frame)

def nd2frame(A, i):
    """
    Given a NumPy nd-array, get the ith frame
    and return it in the format for use with TensorFlow
    image processing methods.
    """
    a = A[i,:,:]
    a = a[...,np.newaxis]
    return a

def weight_variable(shape):
    """
    Create a weight variable with dimensions specified by shape
    and initialized to values drawn from a Gaussian distribution
    with standard deviation of 0.1.
    """
    initial = tf.truncated_normal(shape, stdev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Create a bias variable with dimensions specified by shape
    and initialized to 0.1.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def convolve(x, W):
    """
    Specifies a convolution operation with a kernel specified by
    a weight matrix W (which is what we're learning) and some input x.
    """
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    """
    Specifies a max pooling operation on the input x.
    """
    return tf.nn.max_pool(x, k_size = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
