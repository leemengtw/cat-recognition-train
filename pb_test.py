import numpy as np
import tensorflow as tf
from utils import load_image_dataset


def main():
    features, _ = load_image_dataset()

    # Load graph with metagraph and ckpt
    new_saver = tf.train.import_meta_graph('models/model.ckpt.meta')
    with tf.Session() as sess:
        new_saver.restore(sess, 'models/model.ckpt')
        # Graph loaded; Prediction.
        pred_d = []
        for i in range(64):
            pred_d.append(sess.run('tf_new_y_pred:0',
                          {'tf_new_X:0': features[i:(i + 1)]}))
        pred_d = np.concatenate(pred_d)

    tf.reset_default_graph()  # Clean graph

    # Load graph with frozen graph
    with tf.gfile.GFile('models/frozen.pb', 'rb') as f:
        gd = tf.GraphDef()
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    # Graph loaded; Prediction.
    with tf.Session() as sess:
        pred_f = []
        for i in range(64):
            pred_f.append(sess.run('tf_new_y_pred:0',
                          {'tf_new_X:0': features[i:(i + 1)]}))
        pred_f = np.concatenate(pred_f)
    print(np.all(pred_d == pred_f))
    # Out: True
    return pred_f


if __name__ == '__main__':
    main()
