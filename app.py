import os
import boto3
import json
from flask import Flask, flash, request, json,\
        render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from predict import Predictor
from settings import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMAGE_INFO_JSON, IS_DEBUG

# AWS settings, ignore this part if test locally
try:
    from aws_settings import S3_BUCKET_NAME, S3_BUCKET_BASE_URL,\
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    SAVE_INFO_ON_AWS = True
except ImportError:
    SAVE_INFO_ON_AWS = False

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.secret_key = 'some_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

predictor = Predictor()

# used for rendering after feedback
CUR_PROB = None
CUR_FILENAME = None

def init_image_info():
    """Init settings.IMAGE_INFO_JSON using file stored on S3 for
    warm start.
    """
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if SAVE_INFO_ON_AWS:
        client = boto3.client('s3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        try:
            res = client.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=IMAGE_INFO_JSON\
                .replace(app.config['UPLOAD_FOLDER'], '').replace('/', '')
            )
            image_info = json.loads(res['Body'].read().decode('utf-8'))

            # merge local result
            if os.path.exists(IMAGE_INFO_JSON):
                with open(IMAGE_INFO_JSON, 'r') as f:
                    local_info = json.load(f)

            for k, v in local_info.items():
                if k not in image_info:
                    image_info[k] = v
                elif k in image_info and v.get('label', '') != 'unknown':
                    image_info[k] = v


            # initialize local image_info with S3 version
            with open(IMAGE_INFO_JSON, 'w') as f:
                json.dump(image_info, f, indent=4)

        except:
            # when there is no IMAGE_INFO_JSON on S3
            # just initialize local file and upload later
            pass


@app.route('/', methods=['GET', 'POST'])
def make_prediction():
    """View that receive images and render predictions
    """
    global CUR_PROB
    global CUR_FILENAME

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
            # security concerns
            filename = secure_filename(file.filename)

            # save image for prediction and rendering
            file_path = save_image(file, filename)

            # get prediction
            prob = predictor.predict(file_path)

            # save image info
            save_image_info(filename, prob)


            # keep record of current prediction for later rendering
            # after getting user feedback
            CUR_PROB = prob
            CUR_FILENAME = filename
            # get information of gallery
            images, cur_accuracy, num_stored_images = get_stat_of_recent_images()

            return render_template(
                    'index.html',
                    cat_prob=float('{:.1f}'.format(prob * 100)),
                    dog_prob=float('{:.1f}'.format((1 - prob) * 100)),
                    cur_image_path=file_path,
                    images=images,
                    num_stored_images=num_stored_images,
                    cur_accuracy=cur_accuracy,
                    show_feedback=True)

    # get information of gallery when receive GET request
    images, cur_accuracy, num_stored_images = get_stat_of_recent_images()
    return render_template(
            'index.html', images=images, cur_accuracy=cur_accuracy,
            num_stored_images=num_stored_images)


@app.route('/feedback', methods=['POST'])
def save_user_feedback():
    """Save user feedback of current prediction"""
    global CUR_FILENAME
    global CUR_PROB
    label = request.form['label']

    print('CUR_FILENAME: {}, CUR_PROB: {}'.format(CUR_FILENAME, CUR_PROB))

    init_image_info()

    if CUR_FILENAME:
        # save user feedback in file
        with open(IMAGE_INFO_JSON, 'r') as f:
            image_info = json.load(f)
            image_info[CUR_FILENAME]['label'] = label
        with open(IMAGE_INFO_JSON, 'w') as f:
            json.dump(image_info, f, indent=4)

        if SAVE_INFO_ON_AWS:
            save_image_info_on_s3(image_info)


    # get information of gallery
    images, cur_accuracy, num_stored_images = get_stat_of_recent_images()


    return render_template(
            'index.html',
            prob=float('{:.1f}'.format(CUR_PROB * 100)) if CUR_PROB else 0,
            cur_image_path=uploaded_image_path(CUR_FILENAME),
            images=images,
            num_stored_images=num_stored_images,
            cur_accuracy=cur_accuracy,
            show_thankyou=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """generate url for user uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def allowed_file(filename):
    """Check whether a uploaded file is valid and allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uploaded_image_path(filename):
    """generate file path for user uploaded image"""
    return '/'.join((app.config['UPLOAD_FOLDER'], filename))


def get_stat_of_recent_images(num_images=300):
    """Return information of recent uploaded images for galley rendering

    Parameters
    ----------
    num_images: int
        number of images to show at once
    Returns
    -------
    image_stats: list of dicts representing images in last modified order
        path: str
        label: str
        pred: str
        cat_prob: int
        dog_prob: int

    cur_accuracy: float
    num_stored_images: int
        indepenent of num_images param, the total number of images available

    """
    folder = app.config['UPLOAD_FOLDER']


    # get list of last modified images
    # exclude .json file and files start with .
    files = ['/'.join((folder, file)) \
        for file in os.listdir(folder) if ('json' not in file) \
        and not (file.startswith('.')) ]

    # list of tuples (file_path, timestamp)
    last_modified_files = [(file, os.path.getmtime(file)) for file in files]
    last_modified_files = sorted(last_modified_files,
                            key=lambda t: t[1], reverse=True)
    num_stored_images = len(last_modified_files)



    # read in image info
    with open(IMAGE_INFO_JSON, 'r') as f:
        info = json.load(f)

    # build info for images
    image_stats = []
    for i, f in enumerate(last_modified_files):
        # set limit for rendering pictures
        if i > num_images: break

        path, filename = f[0], f[0].replace(folder, '').replace('/', '')
        cur_image_info = info.get(filename, {})

        print()
        print('path: {}'.format(path))
        print('filename: {}'.format(filename))
        print('cur_image_info: {}'.format(cur_image_info))
        print()

        prob = cur_image_info.get('prob', 0)
        image = {
            'path': path,
            'label': cur_image_info.get('label', 'unknown'),
            'pred': cur_image_info.get('pred', 'dog'),
            'cat_prob': int(prob * 100),
            'dog_prob': int((1 - prob) * 100),
        }
        image_stats.append(image)

    # comput current accuracy if labels available
    total, correct = 0, 0
    for image in image_stats:
        if image['label'] != 'unknown':
            total += 1
            if image['label'] == image['pred']:
                correct += 1

    try:
        cur_accuracy = float('{:.3f}'.format(correct / float(total)))
    except ZeroDivisionError:
        cur_accuracy = 0

    # print(image_stats)
    # print(cur_accuracy)

    return image_stats, cur_accuracy, num_stored_images


def save_image(file, filename):
    """Save current images to setting.UPLOAD_FOLDER and return the
    corresponding file path. Also save the image to AWS S3 if aws information
    is available.

    Parameters
    ----------
    file: werkzeug.datastructures.FileStorage
    filename: str
        pure file name with extension

    Returns
    -------
    file_path: str
        the complete path of the saved image

    """

    # create folder for storing images if not exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # save image locally
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # save image to S3
    if SAVE_INFO_ON_AWS:
        client = boto3.client('s3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        file.seek(0)
        client.put_object(
            Body=file.read(),
            Bucket=S3_BUCKET_NAME,
            Key=filename)

    return file_path


def save_image_info(filename, prob):
    """Save predicted result of the image in a json file locally.
    Also save the json to AWS S3 if aws information is available.

    Parameters
    ----------
    filename: str
        pure file name with extension
    prob: float
        the probability of the image being cat
    """

    # save prediction info locally
    with open(IMAGE_INFO_JSON, 'r') as f:
        image_info = json.load(f)
        image_info[filename] = {
            'prob': float(prob),
            'y_pred': 1 if prob > 0.5 else 0,
            'pred': 'cat' if prob > 0.5 else 'dog',
            'label': 'unknown'
        }
    with open(IMAGE_INFO_JSON, 'w') as f:
        json.dump(image_info, f, indent=4)

    # save image info to S3
    if SAVE_INFO_ON_AWS:
        save_image_info_on_s3(image_info)


def save_image_info_on_s3(image_info):
    """Save the json file containing image info on S3

    Parameters
    ----------
    image_info: dict
    """
    client = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    client.put_object(
        Body=json.dumps(image_info, indent=4),
        Bucket=S3_BUCKET_NAME,
        Key=IMAGE_INFO_JSON\
        .replace(app.config['UPLOAD_FOLDER'], '').replace('/', ''))





@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    pass
    # application.run(host='0.0.0.0', port=5602)
    app.run(host='0.0.0.0', port=5000, debug=IS_DEBUG)
