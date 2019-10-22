import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from dog import inception_predict_breed, setup_cnn

UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cnn_model = setup_cnn()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    # Initialize
    global cnn_model

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
            predicted_breed = inception_predict_breed(new_filename, cnn_model)

            return redirect(url_for('index',
                                    filename=new_filename,
                                    breed=predicted_breed))

    return render_template('index.html')

def main():
    app.run(debug=False, threaded=False)

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     inception_model = setup_cnn()
#     filename = r'images/user_images_cat.jpg'
#     x = inception_predict_breed(filename, inception_model)
#     print(x)