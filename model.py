import csv
import cv2
import numpy as np

# read and store lines from driving_log.csv
lines = []
with open('./data/input/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # for each line, extract the path to the camera image
    # but remember that path was recorded on the local machine
    # since I am on the AWS instance
    for line in reader:
        lines.append(line)
        
# update camera image path, so it is valid on the AWS instance
images = []
steering_measurements = []
for line in lines:
    source_path = line[0]
    # an easy way to update the path is to split the path on it's
    # slashes and then extract the final token, which is the filename
    filename = source_path.split('/')[-1]
    # then I can add that filename to the end of the path to the IMG
    # directory here on the AWS instance
    current_path = './data/input/IMG/' + filename
    # once I have the current path, I can use opencv to load the image
    image = cv2.imread(current_path)
    # append loaded image to images list
    images.append(image)
    
    # I can do something similar for my steering measurements, which
    # will serve as my output labels, it'll be easier to load the steering
    # measurements because there are no paths or images to handle
    # extract the 4th token from the csv line, then cast it as float
    steering_measurement = float(line[3])
    steering_measurements.append(steering_measurement)
    
# now that I've loaded the images and steering measurements,
# I am going to convert them to numpy arrays since that is the format
# keras requires
X_train = np.array(images)
y_train = np.array(steering_measurements)

# next I am going to build the most basic network possible just to make sure everything is working
# this single output node will predict my steering angle, which makes this
# a regression network, so I don't have to apply an activation function

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160, 320, 3)))
model.add(Dense(1))

# with the network constructed, I will compile the model
# for the loss function, I will use mean squared error (mse)
# What I want to do is minimize the error between the steering
# measurement that the network predicts and the ground truth steering
# measurement. mse is a good loss function for this
model.compile(loss='mse', optimizer='adam')

# once the model is compiled, I will train it with the feature and label
# arrays I just built. I'll also shuffle the data and split off 20% of
# the data to use for a validation set. I set epochs to 7 since I saw
# with 10 epochs (keras default) that validation loss decreases with just 7,
# then increases. Thus, at 10 epochs, I may have been overfitting training data.
# Hence, at 7 epochs, the validation loss decreases for almost all the epochs.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

# finally I'll save the trained model, so later I can download it onto my 
# local machine and see how well it works for driving the simulator
model.save('model.h5')
