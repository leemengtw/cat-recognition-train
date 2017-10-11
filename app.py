import os
from flask import Flask, flash, request, json,\
        render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from predict import Predictor

BUCKET_NAME = 'machine-learning-models-dev'
MODEL_FILE_NAME = 'purchase-boost/random_forest_v1.pkl'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME.split('/')[-1]

UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.secret_key = 'some_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

predictor = Predictor()


@app.route('/', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # security concerns

            # save image locally
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # get prediction
            prob = predictor.predict(file_path)

            return render_template(
                    'index.html',
                    prob=float('{:.1f}'.format(prob * 100)),
                    cur_image_path=uploaded_image_path(filename))

    return render_template('index.html')


def allowed_file(filename):
    """Check whether a uploaded file is valid and allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uploaded_image_path(filename):
    """generate file path for user uploaded image"""
    return '/'.join((app.config['UPLOAD_FOLDER'], filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """generate url for user uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/image-test', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # security concerns
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/test', methods=['POST'])
def index():
    payload = json.loads(request.get_data().decode('utf-8'))
    prediction = predict(payload['data'])
    return json.dumps(prediction)


def load_model():
    """Load model stored in S3 for making prediction.
    Make sure to update newest model for correct predictions.
    If model is downloaded in previous execution, use existing model.

    Return:
    model object
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)

    if not os.path.isfile(MODEL_LOCAL_PATH):
        bucket.download_file(MODEL_FILE_NAME, MODEL_LOCAL_PATH)

    return joblib.load(MODEL_LOCAL_PATH)


def predict(data):
    """Use loaded model for making prediction and reture the result.

    Parameters:
    data: List of test instances with all ordered feature values
        [[1, 1, 0, ...]]

    Return:
        result: Dict
        {
            'purchase_probability': 0.46,
            'purchasable': 0
        }
    """
    model = load_model()
    prob = model.predict_proba(data)
    result = {
        'purchase_probability': float(prob[0][1]),
        'purchasable': int(model.predict(data)[0])
    }

    return result


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    pass
    # application.run(host='0.0.0.0', port=5602)
    app.run(host='0.0.0.0', port=5000, debug=True)
