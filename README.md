# Udacity Data Scientist Nanodegree -- Dog Breed Classifier

## Project Overview:
This is the dog breed classifier project. This project uses Convolutional Neural Networks (CNNs). In this project, I built a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

## Web Application
I decided to build a web application instead of a blog post. I was inspired by the web application in the Disaster Response Pipeline project. I will provide the user with an input button to browse for an image that would be uploaded to the site. After upload, the neural network will be called with that provided image. The predicted breed will be fed back to the user through a flask jinja template.

The web application is split into two main files:
* dog.py
    * Refactoring of the code in the jupyter notebook exercise into a self contained python file
* flask_route.py
    * Flask web application that utilizes the standalone python code above into a web application

During the file upload process, only image file extensions are allowed such as png, jpg, and gif. I also made use of secure_filename() utility function which renames a file if it contains special characters or spaces.

I ran into a few challenges trying to productionize the cnn into a flask application. The first issue was deciding where to put the setup of the neural network. At first, I put it in the main of the file but when the flask application runs, the variables were not scoped properly. Then I tried putting it inside the index route but that meant initalizing the whole neural network everytime the user uploads an image. I ended up initializing the network next to where the flask application was initialized. The thing to remember is when I reference the variable later in the index routes, I have to call "global" in order for python to reference the variable I've already initialized.

The second issue I ran into was the flask site would freeze and crash when I ran the prediction method: inception_predict_breed(). Debugging this issue was nearly impossible because no proper and reasonable error was actually thrown. After some research online, I realized that flask must be run in a non-multi-threaded mode and this fixed the issue. This however, will limit the application to only one user at a time. 

### Files:
* dog_app.ipynb
    * Jupyter notebook containing the code used to work through the project
* dog.py
    * Refactoring of the code in the jupyter notebook above into a self contained python file
* flask_route.py
    * Flask web application that utilizes the standalone python code above into a web application
    
### To run the code:
- jupyter notebook dog_app.ipynb
- python flask_route.py
