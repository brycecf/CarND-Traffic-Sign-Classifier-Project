# **Classify Traffic Signs** 

## Project Goal

---

The goal of this project was to design and implement a convolutional neural network that can identify 43 different types of German traffic signs. It was implemented using Google's [TensorFlow](https://www.tensorflow.org/versions/r1.5/get_started/) and its [Keras package](https://www.tensorflow.org/versions/r1.5/api_docs/python/tf/keras).


[//]: # (Image References)

[training_images]: ./documentation_images/training_images.png "15 traffic signs from the training data"
[valid_dist]: ./documentation_images/valid_distribution.png "Class distributions for the training and validation data sets"
[small_dist]: ./documentation_images/small_distribution.png "Classes that are not prevalant in the data"
[vgg16]: ./documentation_images/imagenet_vgg16.png "VGG16 architecture visualization"
[orig_signs]: ./documentation_images/eight_signs.png "Eight German traffic signs"
[results]: ./documentation_images/signs.png "Eight German traffic sign predictions"
[probs]: ./documentation_images/probs.png "Top 5 probabilities for each German sign"

---

### Data Set Summary & Exploration


Every image in this variation of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) consists of three channels (RGB) and has a size of 32x32 pixels. There are 51,839 images total, which are divided into three different sets for training, validation, and testing.

| Data Set       | # of Images   |
| :------------- |:-------------:|
| Training       | 34,799        |
| Validation     | 4,410         |   
| Test           | 12,630        |

An image can belong to one of 43 different types of traffic signs. To provide a better idea of what the traffic sign types, here is a snippet of fifteen training images (the number below an image is the class ID).

![alt text][training_images]

Next, by looking at the class distributions, we can clearly see it is not uniform in the training data. However, when we look at the validation data, we can clearly see that the sampling distribution mirrors that of the training data.

![alt text][valid_dist]

Thus, no data augmentation was pursued since the training data would be able to accurately reflect the validation set. Nevertheless, I was interested in seeing what types of images were not prevalent in the data (as shown below). 

![alt text][small_dist]

That these traffic signs do not form a large portion of the data set is not suprising, given that some of the most common traffic signs are for speed limits.

---

### Model Architecture

![alt text][vgg16]

The model architecture I designed was inspired by the famous VGG16 deep neural network architecture from the ILSVRC-2014 competition. In general, VGG16 inspired architectures can be characterized as placing greater emphasis on layer depth (as the image moves forward through the network blocks) than width. Furthermore, within each network block, you typically see multiple convolutions or dense layers with the same dimensions.

In the case of my architecture, I did not follow this "consecutive" pattern in the first two layers and reduced the overall size of the neural network since I trained the model on a far smaller corpus than ImageNet, which the complete VGG16 architecture is trained upon.  I did include two copies of the convolutional layers and two of the dense layers within the last two blocks since I wanted the training to focus on the higher-level features that distinguish one sign from another.  The architecture also utilizes batch normalization to improve training speed and reduce overfitting.

**Final model architecture:**

| Layer         		|     Description	        					| Activation | Output Dimensions |
| :-------------------- |:---------------------------------------------:|:----------:|:-----------------:| 
| Input         		| 32x32x3 RGB image   							| RELU       | 32x32x3           |
| 3x3 Convolution     	| 1x1 stride w/ same padding                	|            | 32x32x64          |
| Batch norm			| Momentum: 0.9, Epsilon: 0.001					|            | 32x32x64          |
| 2x2 Max pooling	    | 2x2 stride w/ valid padding                   |            | 16x16x64          |
| 3x3 Convolution	    | 1x1 stride w/ same padding      				| RELU       | 16x16x128         |
| Batch norm			| Momentum: 0.9, Epsilon: 0.001					|            | 16x16x128         |
| 2x2 Max pooling	    | 2x2 stride w/ valid padding                   |            | 8x8x128           |
| 3x3 Convolution	    | 1x1 stride w/ same padding      				| RELU       | 8x8x256           |
| Batch norm			| Momentum: 0.9, Epsilon: 0.001					|            | 8x8x256           |
| 2x2 Max pooling	    | 2x2 stride w/ valid padding                   |            | 4x4x256           |
| Flatten        	    |                                               |            | 4096              |
| Fully-connected  	    | 256 units                                     | RELU       | 256               |
| Batch norm			| Momentum: 0.9, Epsilon: 0.001					|            | 256               |
| Fully-connected  	    | 256 units                                     | RELU       | 256               |
| Batch norm			| Momentum: 0.9, Epsilon: 0.001					|            | 256               |
| Fully-connected  	    | 43 units                                      | SOFTMAX    | 43                |

---

### Model Training

#### Preprocessing Inputs
Initially, I tried normalizing the data to have a a zero mean and equal variance. However, when I tried just normalizing the variance, it actually had better performance, so I used that technique.

#### Training Process
I used the Adam optimizer since it is a highly recommended alternative to stochastic gradient descent (and the other alternatives). By using Adam, I am able to take advantage of both benefits associated with AdaGrad and RMSProp. This should then greatly improve the training time and performance of my model.

For the training batch size, I used a minibatch of 64. If I used a minibatch size of 256, my model's performance actually deteriorated (although it was faster to train). Presumably, this is because the less noise of the larger batch size actually caused the model to jump over some local minima.

I ran ten different models (with the same architecture) with ten different random seeds for the weight initalizations. Then, I averaged the weights of each model to arrive at a more consistent result. When I use the average weights, the model's accuracy outperforms any individual model result. 

Each model was trained for a maximum of ten epochs, although this never happened as I utilized early stopping in an effort to prevent overfitting on the training data (although overfitting does persist).

Ultimately, the training process resulted in **97.96%** accuracy on the validation set. In the original 2011 competition where this data set was utilized, this model would have come in fourth place (under multi-scale CNNs) and 0.99% worse than human-level performance. It should be noted that I did not perform any data augmentation when training my model (since it is not necessary for my purposes), so it should be able to easily match these results.

---

### Performance on New Images from the Internet

I collected eight German traffic sign images from [Wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany)(since it seemed that nearly all the Google image results contained watermarks).

![alt text][orig_signs]

As these images are much higher resolution than 32x32, I resized them using the [OpenCV](https://docs.opencv.org/2.4/index.html) package. Running the results through my model resulted in **100%** accuracy. Given that I did not perform any data augmentation, I am impressed that the model was able to correctly identify signs that had few images in the training data and could conceivably be confused with very similar looking signs.

![alt text][results]

Furthermore, when we look at the predictions, we can see that the model has near perfect confidence for nearly all images. The obvious exception is for *42 - End of no passing by vehicles over 3.5 metric tons*, but its second prediction is for *41 - End of no passing*, so it is easy to see why it has longer confidence given the similarity of the two signs. If I used data augmentation, this would likely further improve the robustness of the model.

![alt text][probs]