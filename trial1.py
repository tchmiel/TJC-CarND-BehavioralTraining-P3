import csv
import cv2
import numpy as np

line = []
with open('/data/P3Data_1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

center_images=[]
measurements = []
for line in lines:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = '/data/P3Data_1/IMG/' + filename
		center_image = cv2.imread(current_path)
		center_images.append(center_image)
		measurement = float(line[3])
		measurements.append(measurement)

X_train = np.array(center_images)
y_train = np.array(measurements)

from keras.models = import Sequential
from keras.layers = import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
