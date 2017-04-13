import tensorflow as tf
import numpy as np

from normal_approximation import *

sess = tf.InteractiveSession()

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
    initial = tf.truncated_normal(shape, stddev=0.1)
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

def upsample(x, shape):
    """
    Specifies a part of an unconvolution operation defined by
    bilinear upsampling. This should later be followed by a ReLU
    and a convolution.
    """
    return tf.image.resize_bilinear(x, shape)

def max_pool(x):
    """
    Specifies a 2x2 max pooling operation on the input x.
    """
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

"""
Construction of the computation graph.
"""

x = tf.placeholder(tf.float32, shape = [None, 1024])
y_ = tf.placeholder(tf.float32, shape = [None, 1024])

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

# Format the input array as a TensorFlow image array.
x_image = tf.image.per_image_standardization(tf.reshape(x, [32, 32, 1])) * 100
x_stdimg = tf.reshape(x_image, [-1, 32, 32, 1])

h_conv1 = tf.nn.relu(convolve(x_stdimg, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(convolve(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# Note that ''middle'' refers to the connecting fully connected layer,
# though this is actually implemented as another convolution.
W_middle = weight_variable([3, 3, 64, 256])
b_middle = bias_variable([256])

h_middle = tf.nn.relu(convolve(h_pool2, W_middle) + b_middle)

W_unconv1 = weight_variable([3, 3, 256, 64])
b_unconv1 = bias_variable([64])

h_unconv1 = convolve(tf.nn.relu(upsample(h_middle, [16, 16])), W_unconv1) + b_unconv1

W_unconv2 = weight_variable([3, 3, 64, 32])
b_unconv2 = bias_variable([32])

h_unconv2 = convolve(tf.nn.relu(upsample(h_unconv1, [32, 32])), W_unconv2) + b_unconv2

# This is the final output convolution, hence no pooling is used.
W_outconv = weight_variable([3, 3, 32, 1])
b_outconv = bias_variable([1])

y_image = tf.nn.relu(convolve(h_unconv2, W_outconv) + b_outconv)
y_png = tf.image.encode_png(tf.cast(tf.reshape(y_image, [32, 32, 1]) * 100, tf.uint8))
y_flat = tf.reshape(y_image, [-1, 1024])

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y_flat))
cross_entropy = tf.reduce_mean(tf.squared_difference(y_flat, tf.reshape(tf.image.per_image_standardization(tf.reshape(y_, [32, 32, 1])), [-1, 1024])))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_flat, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
sess.run(tf.global_variables_initializer())
tf.set_random_seed(413)

for i in range(100000):
    I, I_, num_frames = generate_image()
    if i % 100 == 0:
        print('step {}'.format(i))
    accuracies = []
    for j in range(num_frames):
        if i%100 == 0:
            train_accuracy, loss, png = sess.run([accuracy, cross_entropy, y_png], feed_dict = {x: np.reshape(I[j,:,:], (1, 1024)), y_: np.reshape(I_[j,:,:], (1, 1024))})
            # train_accuracy = accuracy.eval(feed_dict={x: np.reshape(I[j,:,:], (1, 1024)), y_: np.reshape(I_[j,:,:], (1, 1024))})
            print("frame %d: loss %f"%(j, loss))
            accuracies.append(train_accuracy)
            with open('test{}.png'.format(j), 'wb+') as f:
                f.write(png)
        train_step.run(feed_dict={x: np.reshape(I[j,:,:], (1, 1024)), y_: np.reshape(I_[j,:,:], (1, 1024))})
    if i%100 == 0:
        print('=====================================')
