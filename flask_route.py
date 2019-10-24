import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from dog import inception_predict_breed, setup_cnn

# Setup upload folder and file extensions
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Setup flask application and attach the upload folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the Neural network model
cnn_model = setup_cnn()

# Function that checks for allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Reference the neural network model that's already setup globally
    global cnn_model

    if request.method == 'POST':
        print('POST')

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
            # Secure the filename
            filename = secure_filename(file.filename)

            # Setup the filename path
            new_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(new_filename)
            print(new_filename)

            # Predict the dog breed
            predicted_breed = inception_predict_breed(new_filename, cnn_model)
            print(predicted_breed)

            # Setup the display message
            message = "Thank you for uploading file {}. Your predicted dog is {}.".format(filename, predicted_breed)

            # Render the HTML with the message
            return render_template('index.html', message=message, filename=new_filename)

    # Render default html page
    return render_template('index.html')

def main():
    app.run(debug=False, threaded=False)

if __name__ == '__main__':
    main()