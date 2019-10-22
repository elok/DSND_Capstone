import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from extract_bottleneck_features import *
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tqdm import tqdm
from keras.layers import Dropout
from keras.applications.resnet50 import preprocess_input, decode_predictions




from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

inception_model = None

UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def show_img(img_path):
    # Read the file, convert the color, and show the image
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_rgb)
    return imgplot

### and returns the dog breed that is predicted by the model.
def inception_predict_breed(img_path):
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))     # extract bottleneck features
    predicted_vector = inception_model.predict(bottleneck_feature)       # obtain predicted vector
    return dog_names[np.argmax(predicted_vector)]                       # return dog breed that is predicted by the model

def predict_breed(img_path):
    # Display the image
    show_img(img_path)

    # Detect for a human face
    if face_detector(img_path):
        return inception_predict_breed(img_path)
    # Detect for a dog
    elif dog_detector(img_path):
        return inception_predict_breed(img_path)
    else:
        raise Exception('No human or dog found. Thanks for playing.')

def setup_cnn():
    # -------------------------------------------
    # Obtain bottleneck features from another pre-trained CNN.
    # -------------------------------------------
    bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
    train_inception = bottleneck_features['train']
    valid_inception = bottleneck_features['valid']
    test_inception = bottleneck_features['test']

    # -------------------------------------------
    # Define the architecture.
    # -------------------------------------------
    inception_model = Sequential()
    inception_model.add(GlobalAveragePooling2D(input_shape=train_inception.shape[1:]))
    inception_model.add(Dense(500, activation='relu'))
    inception_model.add(Dropout(0.4))
    inception_model.add(Dense(133, activation='softmax'))

    inception_model.summary()

    # -------------------------------------------
    # Compile the model.
    # -------------------------------------------
    inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # -------------------------------------------
    ### TODO: Train the model.
    # -------------------------------------------
    # from keras.callbacks import ModelCheckpoint
    #
    # checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inception.hdf5',
    #                                verbose=1, save_best_only=True)
    #
    # inception_model.fit(train_inception, train_targets,
    #           validation_data=(valid_inception, valid_targets),
    #           epochs=100, batch_size=20, callbacks=[checkpointer], verbose=1)

    ### TODO: Load the model weights with the best validation loss.
    inception_model.load_weights('saved_models/weights.best.inception.hdf5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'fileToUpload' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['fileToUpload']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            new_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(new_filename)

            predicted_breed = inception_predict_breed(new_filename)

            return redirect(url_for('index',
                                    filename=new_filename,
                                    breed=predicted_breed))

    return render_template('index.html')

def main():
    # setup_cnn()
    app.run()

if __name__ == '__main__':
    main()