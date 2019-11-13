import csv
import cv2
import sys
import numpy as np

# loads image from filepath using opencv
def get_image(basepath, filepath):
    # read in images from center, left and right cameras
    source_path = filepath
    # extract filename from filepath using split and check platform
    if sys.platform == 'win32':
        filename = source_path.split('\\')[-1]
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        filename = source_path.split('/')[-1]
    # adds filename to end of path to IMG directory, so platform isn't an issue
    img_path_on_fs = basepath + filename
    # load image using opencv
    image = cv2.imread(img_path_on_fs)
    return image

# read and store from driving_log.csv

# Extract left, center and right camera images along with their 
# associated steering angles line by line from driving_log.csv
# stores driving behavior data into images and steering_measurements list
images = []
steering_measurements = []
with open('./data/input/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # for each line, extract the path to the camera image
    # considering that the local machine filepath, does processing so it can
    # run on AWS or other device if needed
    for row in reader:
        steering_center = float(row[3]) # row, column 3 = steering center angle
        
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # parameter to tune
        # steering left of center, recover back to center
        steering_left = steering_center + correction
        # steering right of center, recover back to center
        steering_right = steering_center - correction
        
        # read in images from center, left and right cameras
        basepath = "./data/input/IMG/"
        image_center = get_image(basepath, row[0])
        image_left = get_image(basepath, row[1])
        image_right = get_image(basepath, row[2])
        
        # insert multiple elements into list
        images.extend([image_center, image_left, image_right])
        steering_measurements.extend([steering_center, steering_left, steering_right])

# Augment the training set and help the model generalize better by flipping the
# images horizontally like a mirror and inverting the steering angles. Now the
# dataset should be more balanced, so the model can learn to steer the car 
# clockwise and counterclockwise on the track.
augmented_images, augmented_steering_measurements = [], []
for image, measurement in zip(images, steering_measurements):
    augmented_images.append(image)
    augmented_steering_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1)) # flip img horizontally
    augmented_steering_measurements.append(measurement*-1.0) # invert steering

# Convert images and steering angles to numpy arrays for keras' required format
X_train = np.array(images)
y_train = np.array(steering_measurements)

# Keras CNN Model Implementation based on Nvidia Self-Driving Car Net Architecture
# Network takes in an image, outputs a steering angle prediction
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Keras CNN Architecture
model = Sequential()
# Layer 1: Data Preprocessing
# Normalize input images to pixel value range [-0.5,+0.5] on all images
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (160, 320, 3)))
# Crop2D layer used to remove top 40 pixels, bottom 30 pixels of image
model.add(Cropping2D(cropping = ((50,20), (0,0))))
# Layer 2: Convolutional. 24 filters, 5 kernel, 5 stride, relu activation function
model.add(Conv2D(24,5,5, subsample = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 3: Convolutional. 36 filters
model.add(Conv2D(36,5,5, subsample = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 4: Convolutional. 48 filters
model.add(Conv2D(48,5,5, subsample = (2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 5: Convolutional. 64 filters
model.add(Conv2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 6: Convolutional. 64 filters
model.add(Conv2D(64,3,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
### Flatten output into a vector
model.add(Flatten())
# Layer 7: Fully Connected
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 8: Fully Connected
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 9: Fully Connected
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))
# Layer 10: Fully Connected
model.add(Dense(1))

# Configure the learning process with adam optimizer and mse loss function
# MSE is a good loss function for helping with minimizing the error between
# the steering angle predicted and the ground truth steering angle
model.compile(loss='mse', optimizer='adam')

# Train the model with the image and steering angle arrays
# Shuffle the data, split it off 20% to use for the validation set
# Set epochs to 4 since validation loss decreases and the model is a powerful CNN
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)

# Save the trained model, so later I can download it onto my 
# local machine and see how well it works for driving the simulator
model.save('model.h5')
