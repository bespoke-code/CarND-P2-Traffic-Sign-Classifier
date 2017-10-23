#**Traffic Sign Recognition** 

##Project Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. 
In the code, the analysis was done using python and numpy methods.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is 32px by 32px, 3 channels (R,G,B)
* The number of unique classes/labels in the data set is 43

A dictionary of all the class names was generated in this step.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing how the images are distributed across classes.

![Data distribution](./data/data-distribution.png)

We can clearly see that the dataset does not have an uniform distribution - the number of images per class differs. 
This means that we may encounter worse prediction results for more rare classes compared to more frequently seen ones, 
as the network will be slightly biased and more confident to predict some of the frequently seen image classes.

###Design and Test a Model Architecture

####1. Image data preprocessing. What techniques were chosen and why
Traffic signs in real life are most easily distinguishable by their shape (category) and the images or symbols 
found on them. 

Since the majority of signs sport a white background and black/red/blue/yellow symbols/marks, I decided 
to strip the colour information from each image and work with grayscale images, since I reckon that the sign's shape and 
symbol shape is far more important (if not crucial) to recognizing the sign compared to colour.

My preprocessing pipeline makes use of numpy and skimage functions. I equalize each image's histogram to fix 
brightness and contrast issues, and to put emphasis on each sign's edges, as I believe they are crucial for good results, 
as I expected the convolutional filters to pick some of them up.

Here is an example of 3 traffic sign images before and after the preprocessing step.

![raw and preprocessed images](/home/andrej/git/CarND-P2-Traffic-Sign-Classifier/data/data-preprocessed.png)

While working on the project I was also considering an alternative preprocessing technique which only normalized each image by 
subtracting the mean from each grayscaled pixel, then dividing by the standard deviation. 

You can see a comparison in the preprocessed image output in the figure above. My reasoning was that the shapes and overall 
image look is better after histogram equalization compared to mean normalization, so I sticked with that. Also, the signs' edges 
and shapes/digits/letters were far more legible using this method, for what it's worth.


####2. Final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		| Description	        					    | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale images, normalized          |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  5x5x16 				    |
| Flatten       	    | outputs 400  									|
| RELU					|												|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropouts				|												|
| Logits        		| outputs 43 class predictions					|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer for optimizing 25 epochs and batch size of 256 images. Learning rate: 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

It took  312.8309726715088  seconds to train the network on my Laptop's i7-6820HQ.

My final model results were:
* training set accuracy of **0.985**
* validation set accuracy of **0.947**
* test set accuracy of **0.921** 

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
 

###Testing the Model on New Images

####1. Five German traffic signs found on the web. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image 1](./data/downloaded-img-2.jpg) 
![image 2](./data/downloaded-img-3.jpg) 
![image 3](./data/downloaded-img-4.jpg)
![image 4](./data/downloaded-img-5.jpg) 
![image 5](./data/downloaded-img-6.jpg)

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 