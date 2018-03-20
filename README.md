## Project: Build a Traffic Sign Recognition Program

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_set.png "Visualization"
[image2]: ./examples/test_set.png "Visualization"
[image3]: ./examples/valid_set.png "Visualization"
[image4]: ./examples/43_signs.png "German Traffic Signs"
[image5]: ./examples/color_set.png "color set"
[image6]: ./examples/gray_set.png "gray set"
[image7]: ./examples/norm_set.png "normalized set"
[image8]: ./examples/tran_sam_1.png "Transformation and augmentation"
[image9]: ./examples/tran_sam_2.png "Transformation and augmentation"
[image10]: ./examples/tran_sam_3.png "Transformation and augmentation"
[image11]: ./examples/test_samp.png "Sample Images"
[image12]: ./examples/gray_samp.png "Sample Images"
[image13]: ./examples/final_result.png "Sample Images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/santhoshpkumar/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in each of the test, train and validation set

![alt text][image1]

![alt text][image2]

![alt text][image3]

It can be observed that the distribution of the sample are similar across all 3 datasets. 

Sample image of each of the 43 sign that are contained in the dataset.

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step of preprocesing, I decided to convert the images to grayscale because my earlier attempt to use the color images and contract addjustment did not produce a better result model. It might eventually get batter if agumented with more samples. Taking lead from earlier project of lane detection it made more sense to use the grayscale which might help in identifying patterns in the singages.

Here is an example of random traffic sign images before and after grayscaling.

![alt text][image5]

![alt text][image6]

I decided to generate additional data because the given data set was small to train the model to get good accuracy. Given that the data set is of images or signs they remain constant and easy to creat copies with soem variations. 

To add more data to the the data set, I used the techinique of just random shifting the image, this mimics the true world where the image might be slightly at different angle as observed. At first I created 3 random shift copeis of the image, but this was still not sufficient and hence bumped the multiplier factory by 4. Each image gets 4 copies of the image with some shift transformations.

Here is an example of an original image and 4 set of augmented images:

![alt text][image8]

![alt text][image9]

![alt text][image10]

The difference between the original data set and the augmented data set is that there is a shift in the image. OpenCV functions are used to get the resulted transform effect.

As a last step, I normalized the image data so that all images follow a mean and the signs dont get distinguised differently. 

Here is the random sample of normalized images.

![alt text][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I followed the original LeNet architecture. Just using the same architecture was giving validation accuracy of 89%. I added a additional convolution layer before flattening the image. The key to the model was to generate more data to train the model, these steps have already been explained. 
I used EPOCH of 10 and pretty much followed as much of the LeNet lab, I randomly picked a batch size and finally settled with 164, not much analysis and experiment was done on that part. I used the AdamOptimizer with a learning rate of 0.00097. I also added dropout as suggested in the lecture with 50% dropout ratio.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.959 
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

Before preprocessing

![alt text][image11]

After preprocessing

![alt text][image12]

The difficult of the sign is the No Vehicles labvel 15, A slight shadow can cause the sign to be classifed as the other spped limit signs or any other sign with round border.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Accuracy

|:---------------------:|:---------------------------------------------:| 

| Speed Limit (30 km/hr)      		| Speed Limit (30 km/hr)   									|   1.000
| Bumpy road     			| Bumpy road 										|   1.000
| Ahead only					| Ahead only											|   1.000
| No vehicles	      		| No vehicles					 				|   0.750
| Go straight or left			| Go straight or left      							| 0.800

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is able to predict most of the images correctly and with 100% confidence. 

![alt text][image13]
