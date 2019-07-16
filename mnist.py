# -*- coding: utf-8 -*-
"""mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K7lZZJ2gHaXHsoVHsHIlmoqqQbuYVJbF
"""

#!/usr/bin/env python

'''
mnist.py - Implemented a CNN using tensorflow to classify the MNIST data set
'''

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


# Load data
dataset = input_data.read_data_sets("/tmp/data/", one_hot = True)
(train_images_sq, train_labels), (test_images_sq, test_labels) = keras.datasets.mnist.load_data()

# reshape images to specify that it's a single channel
train_images = train_images_sq.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images_sq.reshape(test_images.shape[0], 28, 28, 1)

# A bit of preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0


# Building the model 
model = keras.Sequential()
# Add Layer by layer

# Conv layer 1: Input: 28*28 pixel values
#               Filter size = 3*3
#               Stride = 1
#               number of filters 32 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Conv layer 2: Input: Output of layer 1
#               Filter size = 3*3
#               Stride = 1
#               number of filters 64
model.add(Conv2D(64, (3, 3), activation='relu'))

# Max pooling with filter size 2*2
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten since too many dimensions, we only want a classification output
#model.add(Flatten())

# Fully Connected layer: Input: Output after max pooling
#                        Number of nodes = 128
model.add(Dense(128, activation='relu'))

# output layer: input: Output from prev layer
#                10 neurons indicating probabilites of each class
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images,train_labels,epochs=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

# Make predictions
predictions = model.predict(test_images)

print("############Prediction for some of the test images##########")
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_sq[i])
    plt.xlabel(np.argmax(predictions[i]))
plt.show()
