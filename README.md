In this repo will solve image classification problems using Convolutional Neural Network (CNN).

Create a folder called **workshop/cnn** somwehere in your hard drive and write all code there.

# Workshop - Basic CNN
In this workshop we will recongnize the classic MNIST data set. These are small 28x28 greyscale images of hand written digits. They need to be classfied from 0 to 9.

We keep the problem simple so that we can focus on the CNN architecture.

## View the Images
Copy ``view_mnist.py`` from the solution folder to **workshop/cnn**. Then run it.

```
python3 view_mnist.py
```

This will give you an idea about the images.

## Load the Data
In **workshop/cnn** folder crearte a file called ``simple_cnn.py``. Add this code to load the image data.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
```

Image data ``train_images`` and ``test_images`` will have the dimensions ``mx28x28``. Where ``m`` is the number of samples. CNN requires each image to have depth. For RGB color images the depth is 3. Here, because the images are greyscale, the depth diminsion is not created by the image loader. We must reshape the data to add that.

Add these lines.

```python
image_width = train_images.shape[1]
image_height = train_images.shape[2]

#Reshape to add a depth of 1
train_images = train_images.reshape(
    (train_images.shape[0], image_width, image_height, 1))
test_images = test_images.reshape(
    (test_images.shape[0], image_width, image_height, 1))
```

## Create the Model
At this time Tensorflow doesn't have any estimator for CNN. We should use Keras to quickly build the model. Our model will have this architecture.

- Input layer has the same dimension as each image. That is 28x28x1. 
- Convolution layer #1 will have a depth of 5. We will convolve using a window of size 3x3. Activation for all CNN layer will use ``relu()`` function.
- We will reduce the image size using a max pool layer. It will use a window of 2x2 and output the highest pixel value. Stride will be 2 pixels. This will output 5 images of size 14x14 (one image for each depth of the previous CNN layer).
- Convolution layer #2 will have a depth of 10. Convolution window size 3x3.
- Subsample the image again using a max pool layer.
- Convolution layer #3 will have a depth of 10. Convolution window size 3x3.
- We will flatten the output from all neurons from the CNN layer #3 into a vecor.
- We will feed that vecor as input to a regular neural network layer consisting of 64 neurons.
- Finally, we will have the output layer. It will have 10 neurons, one for each class. The activation function here will be softmax so that the output of one of the neurons is clearly the maximum. The total output from all 10 neurons will be 1.0.

Add this code. Make sure this jives with the architecture stated above.

```python
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
```

>Note that after the first layer we do not need to specify the shape of the input. This is easily derived by the model by inspecting the previous layer.

# Train and Evaluate
Add this code.

```python
# Start training              
model.fit(train_images, train_labels, epochs=3)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Accuracy:", test_acc)
```

Save changes.

Run the code.

```
python3 simple_cnn.py
```

You should get about 97% accuracy.

# Workshop - See Convolution in Action
It will help to see actually how convolution works. The concept of convolution is very old and goes back to late 1700s. Many image editor filters use convolution. In this workshop we will apply an image blurring convolution. We will use low level Tensorflow API. We will not be doing any training. Instead, we will feed the weights (as well as the image data) as input.

>Convolution takes as input an image and a filter matrix. It outputs another image. In this workshop the output image will be a blurred version of the original image. In a real image recognition problem with deep CNN layers it's not entirely clear what these output images mean. It is thought that the model looks for key features like color blotches, shapes and edges.

Copy ``halftone.png`` from the solution folder to **workshop/cnn**. Feel free to view the image.

## Load the Image
Create a file called ``blurr.py``. Add these lines to create the input image data.

```python
import matplotlib.image as img
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

input_image = img.imread("halftone.png")

image_height = input_image.shape[0]
image_width = input_image.shape[1]

# Add a new dimension for depth (of 1 for grayscale).
# Any CNN layer input data must have depth information.
image_depth = 1

input_image_with_depth = input_image.reshape(
    (image_height, image_width, image_depth))
input_image_list = [
    input_image_with_depth
]
```

## Create the Graph
First define the weights and input features (image). Normally weights are defined as variables since they are learned. Here we will supply the weights that we know will blurr the image.

```python
conv_layer_depth = 1
filter_size = 5

W = tf.placeholder(tf.float32, 
    [filter_size, filter_size, image_depth, conv_layer_depth])
X = tf.placeholder(tf.float32, 
    [None, image_height, image_width, image_depth])
```

A separate filter exists for each image channel of input and output. Hence the shape of the weights tensor is ``filter_size X filter_size X image_depth X conv_layer_depth``.

Now convolve the filter over the input image with a stride of 2 pixels.

```python
stride = 2
conv_layer = tf.nn.conv2d(X, W, strides=[stride, stride], padding='SAME')
```

What should the value of the filter matrix elements be? Our filter has a shape of ``5x5``. Which means there are 25 elements. To blur we need to take an average of the image pixels under the filter window. Which means each element should have a value of 1.0/25.0. Add this code.

```python
# To blur the image we will average all pixels under the filter
each_filter_cell_value = 1.0 / (filter_size * filter_size)
filter_weights = np.full(
    (filter_size, filter_size, image_depth, conv_layer_depth), 
    each_filter_cell_value)
```

## Run the Graph and View Output
Run the graph.

```python
with tf.Session() as sess: 
    output_image_list = sess.run(conv_layer, 
        feed_dict = {X:input_image_list, W:filter_weights})
```

For each input image convolution will produce an output image. Let's view it.

```python
#Grab the first and only output image
output_image = output_image_list[0]
output_image_height = output_image.shape[0]
output_image_width = output_image.shape[1]
# Reshape the image to get rid of the depth so that matplotlib
# can show it.
output_image = output_image.reshape(
    (output_image_height, output_image_width))

# Show before and after images side by side
_, plots = plt.subplots(1, 2)
plots[0].imshow(input_image)
plots[1].imshow(output_image)

plt.show()
```

Save changes. Run the code.

```
python3 blurr.py
```

Make sure that the output image looks like a blurred version of the input.