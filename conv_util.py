import tensorflow as tf
import numpy as np

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))

def conv_layer(input_tensor, depth, filter_size, stride):
    input_depth = input_tensor.get_shape()[3].value
    W = weight_variable([filter_size, filter_size, input_depth, depth])
    b = bias_variable([depth])

    convolution_output = tf.nn.conv2d(input_tensor, W, strides=[1, stride, stride, 1], padding='SAME') + b

    #Apply activation function to each pixel of the output
    return tf.nn.relu(convolution_output)

def max_pool_layer(input_tensor, window_size, stride):
    return tf.nn.max_pool(input_tensor, ksize=[1, window_size, window_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')

def fully_connected_layer(input_tensor, num_neurons):
    input_height = input_tensor.get_shape()[1].value
    input_width = input_tensor.get_shape()[2].value
    input_depth = input_tensor.get_shape()[3].value

    #Build the fully connected layer.
    #This requires flattening the input 3D tensor to a 1D vector
    #Unroll the output from the last pooling layer into a 2D matrix
    #The X is already transposed this way
    Xf = tf.reshape(input_tensor, shape=[-1, input_height * input_width * input_depth])
    Wf = weight_variable([input_height * input_width * input_depth, num_neurons])
    Bf = bias_variable([num_neurons])

    return tf.nn.relu(tf.matmul(Xf, Wf) + Bf)

def readout_layer(input_tensor, num_classes):
    W = weight_variable([input_tensor.get_shape()[1].value, num_classes])
    B = bias_variable([num_classes])

    #Prediction
    logits = tf.matmul(input_tensor, W) + B
    Y_hat = tf.nn.softmax(logits)

    return (logits, Y_hat)

def create_optimizer(logits, predictions, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cost = tf.reduce_mean(cross_entropy)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training step, the learning rate is a placeholder
    graph = tf.train.AdamOptimizer(0.001).minimize(cost)

    return (graph, accuracy)

