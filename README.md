# Udacity Data Scientist Nanodegree - Dog Breed Classifier

## Project Overview

This is the dog breed classifier project. A Convolutional Neural Networks (CNN) is built and trained to determine the breed of a dog. The code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

![dog](https://miro.medium.com/max/888/1*33Xfm4y3O7iTFWY4zrZnNQ.png)

We will create three CNN in total -- first, we will first create a CNN from scratch and see how that performs and then we'll take advantage of a technique called Transfer Learning. Lastly, we will create a CNN with transfer learning as the final step.

We will be using two datasets. The first dataset are dog images which we will split to three parts -- training data set, validation data set, and test dataset. Each dataset is broken into two numpy arrays, the first contains file paths to images and the second contains onehot-encoded classification labels. We also have a list of all dog breed names. The second dataset is of human images. 

Once we have a neural network that we're comfortable with, we will productionize it and deploy it with a flask application. I decided to build a web application instead of a blog post. I was inspired by the web application in the Disaster Response Pipeline project. I will provide the user with an input button to browse for an image that would be uploaded to the site. After upload, the neural network will be called with that provided image. The predicted breed will be fed back to the user through a flask jinja template.

The web application is split into two main files:
* dog.py
    * Refactoring of the code in the jupyter notebook exercise into a self contained python file
* flask_route.py
    * Flask web application that utilizes the standalone python code above into a web application

During the file upload process, only image file extensions are allowed such as png, jpg, and gif. I also made use of secure_filename() utility function which renames a file if it contains special characters or spaces.

The main libraries used for this project are:
* OpenCV and PIL for image processing
* keras and tensorflow for creating the CNN
* matplotlib for plotting

## Analysis

The dataset mentioned above had the following statistics:

* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.
* There are 13233 total human images.

We first built a human face detector. It used a pre-trained face detector from opencv called CascadeClassifier ('haarcascades/haarcascade_frontalface_alt.xml'). The image is then converted from BGR to grayscale and then searches for faces. For each face, a bounding box is added and then the image is converted back to color.

We tested the human face detector with 100 human files and 100 dog files. Human faces were detected on the human files 100% of the time. With the dog files, we detected faces 11% of the time.

Next, we wrote a dog detector using a pre-trained ResNet-50 model. This model has been trained on ImageNet, a very large, very popular dataset used for image classification and otehr vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

Keras CNNs require a 4D array (4D tensor) as input:

![4d](https://www.tutorialspoint.com/tensorflow/images/tensor_data_structure.jpg)

The four dimensions are (nb_samples, rows, columns, channels). nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively. The path_to_tensor() function loads the image, resizes it to a square image that is 224 x 224 pixels, and returns a 4D tensor that looks like:

(1, 244, 224, 3)

![cnn](https://miro.medium.com/max/408/1*jNOmERWFNSDugvcvykMfQQ.jpeg)

Now that the 4D tensor are ready, we have a normalization step in which the mean pixel must be subtracted from every pixel in each image. This is implemented in fuction preprocess_input.

We tested with 100 human files and 100 dog files. 0% images of the first 100 human_files were detected as dog. 100% images of the first 100 dog_files were detected as dog.

Our first CNN model was from scratch. A convolutional neural network is a deep learning algorithm which can take an input like an image, take various aspects or features of an image and assign importance or weights to them. The layers are used to identify high-level and low-level features and then blending or pooling them together. The last step is to add a fully connected layer that returns the final result.

![layers](https://miro.medium.com/max/1255/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

The first network consists of the following layers:

![cnn](https://github.com/elok/DSND_Capstone/raw/377791701997969e07bc4be455fd31e7219bb5c9/images/sample_cnn.png)

The result was an accuracy of 9.8086%

To reduce training time without sacrificing accuracy, we next train a CNN using transfer learning. We used a pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

```
Layer (type)                 Output Shape              Param #   
global_average_pooling2d_1 ( (None, 512)               0         
dense_3 (Dense)              (None, 133)               68229     
Total params: 68,229
Trainable params: 68,229
Non-trainable params: 0
```

The result was an accuracy of 41.9856%

The last CNN we built used a different pre-trained model. I chose the Inception bottleneck feature. It contained the following layers:

```
Layer (type)                 Output Shape              Param #   
global_average_pooling2d_2 ( (None, 2048)              0         
dense_4 (Dense)              (None, 500)               1024500   
dropout_3 (Dropout)          (None, 500)               0         
dense_5 (Dense)              (None, 133)               66633     
Total params: 1,091,133
Trainable params: 1,091,133
Non-trainable params: 0
```

The result of this CNN was an accuracy of 79.4258%

## Conclusion

My expections were the accuracy would be higher. Some thoughts on improving the accuracy would be to try increasing the layers in the network. I'm certainly not familiar with all the images in the dog dataset but using more challenging photos where lighting is poor or the angle of the dogs face is off-center would further improve the training of the network. Lastly, I think other model architectures can explored to increase the accuracy.

I ran into a few challenges trying to productionize the CNN into a flask application. The first issue was deciding where to put the setup of the neural network. At first, I put it in the main of the file but when the flask application runs, the variables were not scoped properly. Then I tried putting it inside the index route but that meant initalizing the whole neural network everytime the user uploads an image. I ended up initializing the network next to where the flask application was initialized. The thing to remember is when I reference the variable later in the index routes, I have to call "global" in order for python to reference the variable I've already initialized.

The second issue I ran into was the flask site would freeze and crash when I ran the prediction method: inception_predict_breed(). Debugging this issue was nearly impossible because no proper and reasonable error was actually thrown. After some research online, I realized that flask must be run in a non-multi-threaded mode and this fixed the issue. This however, will limit the application to only one user at a time. 

### Files
* dog_app.ipynb
    * Jupyter notebook containing the code used to work through the project
* dog.py
    * Refactoring of the code in the jupyter notebook above into a self contained python file
* flask_route.py
    * Flask web application that utilizes the standalone python code above into a web application
    
### To run the code:
- jupyter notebook dog_app.ipynb
- python flask_route.py
