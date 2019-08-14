
import tensorflow as tf
import numpy as np
import conv_util

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.compat.v1.keras.datasets.cifar10.load_data()
    
    #Feature data x_train and x_test has this dimension:
    #(m, image_height, image_width, 3)
    #Where m is the number of samples
    #Each image already has a depth of 3. We don't need to add depth

    #One hot encode label data
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def build_model(image_height, image_width, image_depth, num_classes):
    X = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth])
    Y_ = tf.placeholder(tf.float32, [None, num_classes])

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 40  # first convolutional layer output depth
    L = 20  # second convolutional layer output depth
    M = 10  # third convolutional layer
    N = 400  # Count of neurons in fully connected layer

    # The model
    Y1 = conv_util.conv_layer(X, K, 3, 1)
    Y2 = conv_util.conv_layer(Y1, L, 5, 2)
    Y3 = conv_util.conv_layer(Y2, M, 5, 2)
    Y4 = conv_util.fully_connected_layer(Y3, N)
    logits, Y = conv_util.readout_layer(Y4, num_classes)

    optimizer, learning_rate, accuracy = conv_util.create_optimizer(logits, Y, Y_)

    return (X, Y_, accuracy, Y, optimizer, learning_rate)

def train():
    (x_train, y_train), _ = load_data()

    #Use the first image to get its dimensions
    image = x_train[0]
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_depth = image.shape[2]

    #Use the first label to get the number of classes.
    num_classes = y_train[0].shape[1]

    X, Y_, accuracy, Y, optimizer, learning_rate = build_model(image_height, image_width, image_depth, num_classes)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())

        num_epochs = 5
        batch_size = 150 
        batch_X = np.array_split(x_train, batch_size)
        batch_Y = np.array_split(y_train, batch_size)

        for epoch in range(0, num_epochs):
            for batch in range(0, len(batch_X)):
                sess.run(optimizer, {X: batch_X[batch], Y_: batch_Y[batch], learning_rate: 0.001})
                a = sess.run(accuracy, {X: batch_X[batch], Y_: batch_Y[batch]})
                print("Accuracy:", a)

train()