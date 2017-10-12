import glob
import os
import numpy as np
import tensorflow as tf
from utils import read_image_and_resize
from settings import META_PATH, SAVE_PATH, PB_PATH,\
        TRAIN_X_MEAN_NPY, TRAIN_X_STD_NPY


class Predictor():
    """ A session wrapper which predicts catness given an image.

    Argument:
        size: desired image resize target.
    """

    def __init__(self, size=(64, 64)):
        self.size = size
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.train_mean = np.load(TRAIN_X_MEAN_NPY)
        self.train_std = np.load(TRAIN_X_STD_NPY)
        if os.path.exists(PB_PATH):
            gd = tf.GraphDef()
            with tf.gfile.GFile(PB_PATH, 'rb') as f:
                gd.ParseFromString(f.read())
            with self.graph.as_default():
                tf.import_graph_def(gd, name='')
            self.graph.finalize()
        else:
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(META_PATH)
                saver.restore(self.sess, SAVE_PATH)

    def predict(self, file_path):
        """ Predict catness given an image.

        Argument:
            file_path: path to desired image file.

        Returns:
            prob: probability of catness given the image.
        """
        img = read_image_and_resize(file_path, self.size).astype('float32')
        img = ((img - self.train_mean) / self.train_std)
        pred = self.sess.run(
                'tf_new_y_pred:0',
                {'tf_new_X:0': img.reshape(1, *img.shape)})
        return pred.squeeze()

    def predict_np(self, img):
        """ Predict catness given an image.

        Argument:
            file_path: path to desired image file.

        Returns:
            prob: probability of catness given the image.
        """
        img = ((img.astype('float32') - self.train_mean) / self.train_std)
        pred = self.sess.run(
                'tf_new_y_pred:0',
                {'tf_new_X:0': img.reshape(1, *img.shape)})
        return pred.squeeze()


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


def gen_kaggle_sub():
    xpaths = [os.path.join('datasets', 'test1', '%d.jpg' % i) for i in range(1, 12501)]
    number = [i for i in range(1, 12501)]
    out_f = 'id,label'
    p = Predictor()
    for idx, xpath in zip(number, xpaths):
        img = read_image_and_resize(xpath, size=(64, 64))
        pred = 1 - p.predict_np(img).squeeze()
        print(xpath, pred, end='\r')
        out_f += '\n%d,%f' % (idx, pred)
    with open('submission.csv', 'w') as f:
        f.write(out_f)
    print('\nDone')


if __name__ == '__main__':
    gen_kaggle_sub()
