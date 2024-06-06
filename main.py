# importing libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, datasets
import os

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# cifar10 is a dataset that consists 60,000 coloured images in 10 different classes, each class consisting of 6000 images
# the ten classes are: airplane, automobile, cats, dogs, deer, frog, bird, horse, ship and truck

# normalizing the images (converting 0 - 255 value of pixel to 0 - 1)-> convinient to work with!
training_images, testing_images = training_images / 255, testing_images / 255

# assigning names to the labels as they are numbered
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# for i in  range(16):
#     plt.subplot(4, 4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])

# plt.show()


# choosing 20000 training and 4000 testing  to save time and resources, but accuracy will be reduced
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


# creating model
model = models.Sequential()

# adding input layer with activation func relu
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape= (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# hidden layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#flattening the layer
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

# output layer using softmax activation func to scale all the results so they add up to 1 to get the probability of the answer
model.add(layers.Dense(10, activation='softmax'))

# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# calculating loss and accuracy of the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Accuracy: {accuracy}") 

# saving the model
model.save('imageclassifier.keras')
print('Model Saved!')




# # loading the model
# model = tf.keras.models.load_model('imageclassifier.keras')

# image_number = 1
# # checking if the files exist in the following folder
# while os.path.isfile(f"Images/img{image_number}.jpg"):
#     try:
#         # reading the images 
#         img = cv.imread(f"Images/img{image_number}.jpg")[:,:,0]
#         # converting color scheme from bgr to rgb
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         # predicting the image
#         prediction = model.predict(np.array([img]) / 255)
#         index = np.argmax(prediction)
#         print(f"This Image is probably a {class_names[index]}")
#         plt.imshow(img, cmap=plt.cm.binary)
#         plt.show()
#     except:
#         print("Error!")
#     finally:
#         image_number += 1


