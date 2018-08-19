# https://www.tensorflow.org/tutorials/keras/basic_classification
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib
from tensorflow.contrib.distributions.python.ops.bijectors import inline

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


print(tf.__version__)

# We will use
#   60,000 images to train the network and
#   10,000 images to evaluate/ Test
# how accurately the network learned to classify images.

# Access the Fashion MNIST directly from TensorFlow, just import and load the data:
fashion_mnist = keras.datasets.fashion_mnist

# The train_images and train_labels arrays are the training set
#   —the data the model uses to learn.
# The model is tested against the test set,
#   the test_images, and test_labels arrays.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# see how many entries the models (train and test)
print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(test_labels.shape)

Myarr = 0

'''
print('Label ', train_labels[Myarr])
plt.figure()
plt.imshow(train_images[Myarr])
plt.colorbar()
plt.gca().grid(True)

#this will triger showing the gfx
# plt.show()
'''

# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, cast the datatype of the image components from an integer to a float,
# and divide by 255. Here's the function to preprocess the images:


train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()
