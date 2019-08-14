
import tensorflow as tf
import numpy as np
import conv_util
import sys

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

    # The model
    layer = conv_util.conv_layer(X, 7, 3, 1)
    layer = conv_util.conv_layer(layer, 15, 3, 1)
    layer = conv_util.max_pool_layer(layer, 3, 3)
    layer = conv_util.conv_layer(layer, 7, 3, 1)
    layer = conv_util.max_pool_layer(layer, 3, 3)
    layer = conv_util.conv_layer(layer, 3, 3, 1)
    layer = conv_util.fully_connected_layer(layer, 512)
    logits, Y = conv_util.readout_layer(layer, num_classes)

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
    num_classes = y_train[0].shape[0]

    X, Y_, accuracy, Y, optimizer, learning_rate = build_model(image_height, image_width, image_depth, num_classes)

    with tf.Session() as sess: 
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        num_epochs = 50
        batch_size = 150 
        batch_X = np.array_split(x_train, batch_size)
        batch_Y = np.array_split(y_train, batch_size)

        for epoch in range(0, num_epochs):
            for batch in range(0, len(batch_X)):
                sess.run(optimizer, {X: batch_X[batch], Y_: batch_Y[batch], learning_rate: 0.001})

                if batch % 100 == 0:
                    a = sess.run(accuracy, {X: batch_X[batch], Y_: batch_Y[batch]})
                    print("Accuracy:", a * 100.0, "%")
                    saver.save(sess, "./model.ckpt")

def validate():
    _, (x_test, y_test) = load_data()

    #Use the first image to get its dimensions
    image = x_test[0]
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_depth = image.shape[2]

    #Use the first label to get the number of classes.
    num_classes = y_test[0].shape[0]

    X, Y_, accuracy, Y, optimizer, learning_rate = build_model(image_height, image_width, image_depth, num_classes)

    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # Load the weights and biases
        saver.restore(sess, "./model.ckpt")
        # predictions = sess.run(Y, {X: x_test})
        a = sess.run(accuracy, {X: x_test, Y_: y_test})
        print("Accuracy:", a * 100.0, "%")

if sys.argv[1] == "--train":
    train()
elif sys.argv[1] == "--validate":
    validate()