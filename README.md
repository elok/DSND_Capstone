# Udacity Data Scientist Nanodegree - Dog Breed Classifier

## Project Definition

### Project Overview
This is the dog breed classifier project. A Convolutional Neural Networks (CNN) is built and trained to determine the breed of a dog. The code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

We will create three CNN in total -- first, we will first create a CNN from scratch and see how that performs and then we'll take advantage of a technique called Transfer Learning. Lastly, we will create a CNN with transfer learning as the final step.

We will be using two datasets. The first dataset are dog images which we will split to three parts -- training data set, validation data set, and test dataset. Each dataset is broken into two numpy arrays, the first contains file paths to images and the second contains onehot-encoded classification labels. We also have a list of all dog breed names. The second dataset is of human images. 

Once we have a neural network that we're comfortable with, we will productionize it and deploy it with a flask application. I decided to build a web application instead of a blog post. I was inspired by the web application in the Disaster Response Pipeline project. I will provide the user with an input button to browse for an image that would be uploaded to the site. After upload, the neural network will be called with that provided image. The predicted breed will be fed back to the user through a flask jinja template.

The web application is split into two main files:
* dog.py
    * Refactoring of the code in the jupyter notebook exercise into a self contained python file
* flask_route.py
    * Flask web application that utilizes the standalone python code above into a web application

During the file upload process, only image file extensions are allowed such as png, jpg, and gif. I also made use of secure_filename() utility function which renames a file if it contains special characters or spaces.

### Problem Statement
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

how classifier works

In this project, I built a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

how web app should work

--------------------------------------

### Metrics
Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

The dataset mentioned above had the following statistics:

* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.
* There are 13233 total human images.

We tested the human face detector with 100 human files and 100 dog files. Human faces were detected on the human files 100% of the time. With the dog files, we detected faces 11% of the time.

We wrote a dog detector using a pre-trained ResNet-50 model. We tested with 100 human files and 100 dog files. 0% images of the first 100 human_files were detected as dog. 100% images of the first 100 dog_files were detected as dog.

Our first CNN model was from scratch. The network was consisted of the following layers:

![cnn](https://github.com/elok/DSND_Capstone/raw/377791701997969e07bc4be455fd31e7219bb5c9/images/sample_cnn.png)

The result was an accuracy of 9.8086%

To reduce training time without sacrificing accuracy, we next train a CNN using transfer learning. We used a pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_1 ( (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 133)               68229     
=================================================================
Total params: 68,229
Trainable params: 68,229
Non-trainable params: 0
_________________________________________________________________

The result was an accuracy of 41.9856%

The last CNN we built used a different pre-trained model. I chose the Inception bottleneck feature. It contained the following layers:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 2048)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 500)               1024500   
_________________________________________________________________
dropout_3 (Dropout)          (None, 500)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 133)               66633     
=================================================================
Total params: 1,091,133
Trainable params: 1,091,133
Non-trainable params: 0
_________________________________________________________________

The result of this CNN was an accuracy of 79.4258%

## Analysis

### Data Exploration
Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.


### Data Visualization
Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.


## Conclusion

### Reflection
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

### Improvement
Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.


## Web Application
I ran into a few challenges trying to productionize the cnn into a flask application. The first issue was deciding where to put the setup of the neural network. At first, I put it in the main of the file but when the flask application runs, the variables were not scoped properly. Then I tried putting it inside the index route but that meant initalizing the whole neural network everytime the user uploads an image. I ended up initializing the network next to where the flask application was initialized. The thing to remember is when I reference the variable later in the index routes, I have to call "global" in order for python to reference the variable I've already initialized.

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
