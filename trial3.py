import csv
import cv2
import numpy as np

lines = []
with open('./data/P3Data_3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

center_images=[]
measurements = []
for line in lines:
		source_path = line[0]
		filename = source_path.split('\\')[-1]
		current_path = './data/P3Data_3/IMG/' + filename
		print (current_path)
		center_image = cv2.imread(current_path)
		center_images.append(center_image)
		height, width, channels = center_image.shape
		print (height, width, channels)
		measurement = float(line[3])
		measurements.append(measurement)

X_train = np.array(center_images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model_3.h5')
