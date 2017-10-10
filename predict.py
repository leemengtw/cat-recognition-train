import numpy as np
import tensorflow as tf
from utils import read_image_and_resize, get_rgb_image
from settings import META_PATH, SAVE_PATH, TRAIN_X_MEAN_NPY, TRAIN_X_STD_NPY


def predict_on_new_image(file_path, size=(64, 64)):
    """
    Return the predicted probability of a single image being a 'cat' image

    Parameters
    ----------
    file_path: str
        relative path where the image is stored
    size: final image size after resize operation

    Returns
    -------
    prob: float ranged from (0, 1)
        The probability of the image being a 'cat' image

    """

    # resize image
    image = read_image_and_resize(file_path, size=size, debug=True)

    # normalize image
    train_x_mean = np.load(TRAIN_X_MEAN_NPY)
    train_x_std = np.load(TRAIN_X_STD_NPY)
    image = (image - train_x_mean) / train_x_std

    # reshape image to fit in graph
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])


    # load model and make prediction
    with tf.Session() as sess:
        # Restore variables from disk.
        new_saver = tf.train.import_meta_graph(META_PATH)
        new_saver.restore(sess, SAVE_PATH)
        print("Model restored.")

        # Now, access the op that you want to run.
        graph = tf.get_default_graph()
        tf_new_X = graph.get_tensor_by_name("tf_new_X:0")
        tf_new_y_pred = graph.get_tensor_by_name("tf_new_y_pred:0")

        feed_dict = {tf_new_X: image}

        new_y_pred = sess.run([tf_new_y_pred], feed_dict=feed_dict)

    prob = np.squeeze(new_y_pred)
    return prob
