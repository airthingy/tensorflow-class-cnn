# We will classify handwritten digits as 0 to 9. 
# Source data MNIST

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# train_images and test_images are tensors of dimension m x 28 x 28.
# Where m is the number of samples. Each image is 28x28 pixels.
# Images are greyscale and hence has only one color channel.

image_width = train_images.shape[1]
image_height = train_images.shape[2]

# For CNN the input tensor must have a depth. We will just reshape the input
# data to have 1 depth (since its greyscale). For color images the depth will be 3.
train_images = train_images.reshape(
    (train_images.shape[0], image_width, image_height, 1))
test_images = test_images.reshape(
    (test_images.shape[0], image_width, image_height, 1))

# Create the model
model = models.Sequential()
# Convolution layer. Window size 3x3. Depth of layer 5
model.add(layers.Conv2D(5, (3, 3), activation='relu', input_shape=(image_width, image_height, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Convolution layer. Window size 3x3. Depth of layer 10
model.add(layers.Conv2D(10, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Convolution layer. Window size 3x3. Depth of layer 10
model.add(layers.Conv2D(10, (3, 3), activation='relu'))

# Flatten the last convolution layer into a vecor so we can feed this to
# a regular (dense) neural network layer.
model.add(layers.Flatten())
# Fully connected regular neurons (64 in count). Activation function is relu
model.add(layers.Dense(64, activation='relu'))
# Fully connected regular neurons (10 in count, one for each class). 
# Activation function is softmax since we are doing multi-class classification
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Start training              
model.fit(train_images, train_labels, epochs=3)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Accuracy:", test_acc)