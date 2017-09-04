### **Behavioral Cloning Project** 

#### Thomas J. Chmielenski
#### September 2017

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_architecture]: ./output/final_model_architecture.png "Final Model Architecture"
[image_flipped]: ./output/image_flipped.png "Flipped Image"
[left_center_right_images.png]: ./output/left_center_right_images.png "Left, Center and Right Images"
[model_loss]: ./output/model_loss.png "Model Loss"
[both_steering_angles_histogram]: ./output/both_steering_angles_histogram.png "Both Training Sets: Steering Angles Histogram"

[center_of_lane_driving]: ./output/center_of_lane_driving.png "Center Of Lane Driving"

[steering_correction]: ./output/steering_correction.mp4 "Steering Correction"

## Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
#### Files Submitted & Code Quality

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### 3. Submission code is usable and readable

The `model.p` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `P3_PreprocessModel.ipynb` is a jupyter notebook is an auxilary file that I use to preview image augmentations, as well view plot of the model dataset.

---

### Model Architecture and Training Strategy


##### 1. An appropriate model architecture has been employed

My final model was based off the convolution neural network architecture 
published by the Automonous Vehicle Team at NVIDIA in their paper titled, 
"[End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

I used a Keras Lamba layer to normalize the data (`model.py, line 199`) and cropped the images from 
320x160 pixels down to 320x70 pixels by removing 65 pixels from the top of the image, and 25 pixels from the bottom of the image
 (`model.py, line 200`) 

Similar to the NVIDIA model, I used 5 Convolutional layers (`model.py, lines 201-205`) with RELU activiation layers. However, unlike the NVIDIA model, I used only 3 fully connected layers.  To simplify my model, I removed the fully connected 1164 neuron layer.

My final model architecture is shown here (`model.py, lines 198-213`):
![Final Model Architecture][model_architecture]


##### 2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers between fully connected layer (`model.py, lines 208, 210, 212`) to prevent overfitting the mdodel.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

##### 3. Model parameter tuning

The model used an 'adam' optimizer, using the default learning rate of 0.001 (`model.py, line 220`).

##### 4. Appropriate training data

For training data, I utilized:
 
* the given Udacity's training data
* 2 laps of center lane driving, 
* 1 lap of driving counter clockwise, and 
* 2 laps of driving by second trainer (my teenage daughter)

By utilizing a second person to help with training laps, I believe this helped to 
avoid a bias towards how I would train the car to drive.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

##### 1. Solution Design Approach

My overall strategy for deriving a model architecture was to start with the simplest network 
(as shown in [Section 7. Training Your Network](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/db4960f9-c8d3-4baf-9c7e-678341c1cf3e)), 
then try both the LeNet Architecture [Section 7. Training Your Network](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/0487ab4d-5550-4921-b3f0-5a9f1e4f9e93)),
, but focus on the [NVIDIA architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

Because the NVIDIA architecture is used by the Automonous Vehicle Team at NVIDIA in their self-driving vehicles,
I suspected this would be a good architecture to base my model on.

Before diving into evaluating the LeNet and NVIDIA models, I watched the remaining class videos and 
reviewed the forums and found some good advice from [Paul Heraty's Behavioral Cloning Cheatsheet](https://discussions.udacity.com/t/cant-access-helpful-guide-from-paul-heraty-oct-cohort/217945)
Leveraging what I have learned, I would start by preconfiguring my model by:

*  	Leverage Python Generators to avoid loading all the images into memory as once.  I would start with a batch_size of 32.Generators run 
as a separate process from the main routine, utilitize `yield`, rather than `return` statements. Generators returns, 
but saves the current values of all the generator method's variables.  The generator can be called multiple times, and the code will re-start where it left off right after the yield statement.  Thus, this allows only subsets of images to be load at one time.  

* 	Correct the colorspace of the training images.  `drive.py` feeds the model with RGB images.  For training of the images, we are utilizing the `cv2.imread` method which reads images in the BGR format. I 
will need to convert the images from BGR to RGB before I train the image. 

* Normalize the images, using a lambda  function. Implementating as a lamdda function in the model, 
it will be handle for both the training set as well as in `drive.py`.  
	
* Augment the training data to create additional images.
    * Leverage all three camera images, adding a steering correction angle for camera angle offset.
    * Flip the images, as well as steering measurements to avoid left turn bias. 
    * Crop the images to focus on the areas of the image that really matter, and avoid areas that 
are irrelvant, such as the sky, trees, etc. 

To help visualize the augmentations, I created a Jupyter notebook, [P3_PreprocessModel.ipynb](.\P3_PreprocessModel.ipynb).
All code used in the training the model will be ported over to `model.py`, so reviewers can run the model, if they choose.

In order to gauge how well the model was working, I split 20% of the image and steering angle data 
from the training set, to create a validation set. 

To combat the overfitting, I modified the model so adding three dropout layers with 50% dropout rate 
between each fully connected layer. (`model.py, lines 208, 210, 212`)

Three epochs seems to be the optimal ('model.py', lines 3)

Other experimentation with fine tuning of the model, but the resutls were not sufficient to
warrant further discussion.

With this given model architecture, the vehicel was able to drive autonomoulsy around the track
without leaving the road!




##### 2. Final Model Architecture

The final model architecture (`model.py, lines 198-213`) consisted of a convolution neural network, based on the NVIDIA architecture
utilizing five convulation layers,  followed by 4 fully connected layers, with dropout layouts in between the fully connected layers.

My final model architecture is shown here (`model.py, lines 198-213`):
![Final Model Architecture][model_architecture]

Here is the model loss plot for the final architecture:
![Model Loss][model_loss]

##### 3. Creation of the Training Set & Training Process

I began by creating data by training for 2 laps in the simulator, using the keyboard keys for steering control.
Then training using the simplest network, and viewed the results.  I was expecting my vehicle to behave in a
jittery fashion as the video in [Section 8. Running Your Network](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1ff2cbb5-2d9e-43ad-9424-4546f502fe20).
However, my vehicle drove hard left, continued off the track, and just did a complete circle, and did not
anything like the video.

I found it was very difficult to steer with the mouse and when training with
the keyboard keys, the vehicle was hard to control and it was very easy to swerve in the lane and
even to jump the curb.  Since this was not the desired behavior, I did not use this dataset.

Reviewing the Udacity forums, there was suggestions to utilize a joystick controller to get better training
data with weaving across the lanes.  I also realized that Udacity provided us with a training set (`data.zip`) that we
could utilize to help train the network.  Until I could acquire a joystick, I would use the supplied training set.

I took advantage of the given left and right images, as well as the center image.  I added the left and right images,
and adding in a steering correction of -0.25, 0.25, respectively to the center image's steering angle.
(`model.py, lines 142-151`)  

Here is a sample of the three images from one sample of the training set:
![Left, Center, and Right Images][left_center_right_images.png]


After preprocessing the data, I trained the model using the LeNet Architecture(`model.py, lines 181-193`) as well as the 
NVIDIA architecture(`model.py, lines 198-213`) for comparision.  Both of these models performed better than the initial Simple
architecture(`model.py, lines 173-176`).  The LeNet model left the track soon after it began, but at least it wasn't going
in left handed circles!  With the NVIDIA model, the vehicle left the track as soon as it approached the first left hand curve
before the bridge.  I decided to focus solely on the NVIDIA model and modify this architecture until successful.

The provided Udacity training would not be enough data to successfully train a model and navigate the track.
I utilized a [LogiTech F310 Gamepad](http://gaming.logitech.com/en-us/product/f310-gamepad) to gain more
smoother steering angles.  It was much easier to control the vehicle during training with a joystick than
with either the mouse or keyboard.

For additional training data, I recorded:
 
* 2 laps of center lane driving, 
* 1 lap of driving counter clockwise, and 
* 2 laps of driving by second trainer (my teenage daughter)

For the center lane driving laps, I 

![Center Of Lane Driving][center_of_lane_driving]


#![Steering Correction][steering_correction]


I compared the results of the using just the Udacity training set, 
my training set, and combining all the datasets.  As I would have expected, the combined data improved the model's
performance on the track. Therefore, I decided I could
combine both the Udacity training set as well as my own training set to get the best results.

In order to combine the datasets into one training model, I created a datasets array, and appended each 
dataset.  The Udacity dataset `driving_csv.log` was formatte slightly different then datasets' `driving_csv.log` I created
on my Windows machine.  By cleaning the data ('model.py', lines 13-51), it was easier to work with the left, center, and 
right image filenames as they were now all relatively pathed.

Using this combined dataset, the vehicle successfully navigated the first left turn, over the bridge, the second
left turn, but then cross the right yellow on the one and only right hand turn!

To avoid a bias of driving straight (steering_angle = 0.0), I removed 95% of samples who had 
between -0.01 < steering_angle < 0.01 when loading the dataset ('model.py', lines 11, 36-37)
![both_steering_angles_histogram][both_steering_angles_histogram]

Reviewing my generator code (`model.py, lines 114-168`)), I realized that I never included the `flip_image`
augmentation photo into the training samples. In addition, I found a syntax error in my right_image was not being color corrected.  

Here is an example of a flipped image:
![Flipped Image][image_flipped]


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. 
I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine 
if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by
the validation.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
