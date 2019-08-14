import matplotlib.image as img
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image = img.imread("halftone.png")

# Add a new dimension for depth (of 1 for grayscale)
image_height = image.shape[0]
image_width = image.shape[1]
image_depth = 1

image = image.reshape((image_height, image_width, image_depth))
input_image_list = [
    image
]

conv_layer_depth = 1
filter_size = 5
W = tf.placeholder(tf.float32, [filter_size, filter_size, image_depth, conv_layer_depth])
X = tf.placeholder(tf.float32, [1, image_height, image_width, image_depth])

# Shift 2 pixels after each iteration
stride = 2
conv_layer = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')

# To blur the image we will average all pixels under the filter
each_filter_cell_value = 1.0 / (filter_size * filter_size)
filter_weights = np.full((filter_size, filter_size, image_depth, conv_layer_depth), each_filter_cell_value)

with tf.Session() as sess: 
    output_image_list = sess.run(conv_layer, feed_dict = {X : input_image_list, W : filter_weights})
    print(output_image_list.shape)

# # There was only one input image. So get the first and only
# # output image
# output_image = output_images[0]
# print(output_image.shape)
# output_image_height = output_image.shape[0]
# output_image_width = output_image.shape[1]
# # Reshape the image to get rid of the depth so that matplotlib
# # can show it.
# output_image = output_image.reshape((output_image_height, output_image_width))
# plt.imshow(output_image)
# plt.show()