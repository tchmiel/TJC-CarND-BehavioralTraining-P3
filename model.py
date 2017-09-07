import os
import csv
import numpy as np


TEST_SIZE = 0.20
BATCH_SIZE = 32
NUM_EPOCHS = 3


### load datasets
remove_zero_percentage = 0.95
def relative_image_path (root_data_dir, image_path):
    # converts driving_log.csv image path to be relative image path
    image_path = image_path.strip()
    if image_path.startswith('IMG'):
        # from Udacity driving_csv.log
        relative_path = root_data_dir + image_path
    else:      
        relative = image_path.strip().split('data')[-1]
        relative_path = './data' + relative
        
    relative_path = relative_path.replace('/', os.path.sep).replace('\\', os.path.sep)
    return relative_path
    
def load_dataset(root_data_dir):
    ds_samples = []
    ds_images_removed = 0
    csvfilename = root_data_dir + 'driving_log.csv'
    with open(csvfilename) as csvfile:
        reader = csv.reader(csvfile)
        
        for line in reader:
            # filter out 70% of the near zero steering angles to avoid high bias 
            center_angle = float(line[3])
            prob =  np.random.rand()
            if (abs(center_angle) < 0.01 and prob < remove_zero_percentage):
                ds_images_removed += 1
                continue
            line[0] = relative_image_path(root_data_dir, line[0])
            line[1] = relative_image_path(root_data_dir, line[1])
            line[2] = relative_image_path(root_data_dir, line[2])
            ds_samples.append(line)
    return ds_samples, ds_images_removed

datasets = []
datasets.append("./data/Udacity/")  #Udacity Training Set
datasets.append("./data/CenterLaneDriving/")  
datasets.append("./data/TeenagerDriver/")
datasets.append("./data/CounterClockwise/")

samples = []
images_removed = 0
for ds in datasets:
    ds_samples, ds_images_removed =  load_dataset(ds)
    samples.extend (ds_samples)
    images_removed += ds_images_removed

print("Number of total samples =", len(samples) + images_removed)
print("Number of samples removed =", images_removed)
print("Number of usable samples =", len(samples))


 ###  Utiltiy Functions
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def create_subDir (dir, sub_dir):
    new_path = os.path.join(dir, sub_dir)
    ensure_dir(new_path)
    return (new_path)


### Image Plot utilities
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline



### Image Methods

def saveImage(imageFilename, image):  
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ensure_dir(imageFilename)
    cv2.imwrite(imageFilename,RGB_img)

def displayImage(image):
        plt.imshow(image)
        plt.show()

### Image Methods
def flip_image (image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return (image_flipped, measurement_flipped)

def colorCorrect_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


### Split off 20% of the samples to be a validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = TEST_SIZE)


### generator
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size = BATCH_SIZE):
    steering_correction_factor = 0.25
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                name = batch_sample[0]
                #print(name)
                assert (os.path.isfile(name))
                
                center_image = cv2.imread(name)
                center_image = colorCorrect_image(center_image)
                images.append(center_image)
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                
                #height, width, channels = center_image.shape
                #print (height, width, channels)
                
                # use left and right images, as well, applying a sterring_correction_factor
                # create adjusted steering measurements for the side camera images
                left_image = cv2.imread(name)
                left_image = colorCorrect_image(left_image)
                steering_left_angle = center_angle + steering_correction_factor
                images.append(left_image)
                angles.append(steering_left_angle)

                right_image = cv2.imread(name)
                right_image = colorCorrect_image(right_image)
                steering_right_angle = center_angle - steering_correction_factor
                images.append(right_image)
                angles.append(steering_right_angle)
                
                flipped_image, flipped_angle = flip_image (center_image, center_angle)
                images.append(flipped_image)
                angles.append(flipped_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)



## Model Architecture
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Lambda, Cropping2D, Flatten, Dense, MaxPooling2D, Dropout


# third attempt - NVidia
model = Sequential()
model.add(Lambda(lambda x: ((x / 255.0) - 0.5), input_shape = (160,320,3))) #normalize the data
model.add(Cropping2D(cropping = ((65,25), (0,0))))  # ((PixelsFromTop. FromBottom), (FromLeft,FromRight))
model.add(Convolution2D(24,5,5, border_mode = "valid", subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, border_mode = "valid", subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, border_mode = "valid", subsample = (2,2), activation ='relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


# print out model summary
model.summary()

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, 
            samples_per_epoch = len(train_samples)*4, 
            validation_data = validation_generator, 
            nb_val_samples = len(validation_samples), 
            nb_epoch = NUM_EPOCHS, verbose  = 2)

model.save('model_final.h5')

