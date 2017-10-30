import os
import tensorflow as tf
from scipy.misc import imsave


def main():
    def _parse(path):
        img = tf.image.decode_jpeg(tf.read_file(path))
        # tf.image.decode_png also works
        # both reads both jpeg png
        # then what's the difference wtf
        img = tf.cond(
                tf.less(tf.random_uniform(shape=[], minval=.0, maxval=1.), .5),
                lambda: img,
                lambda: tf.image.flip_up_down(img))
        return img
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, 'dataset_example_img_path')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    paths = tf.constant([os.path.join(root, 'datasets', 'train', 'cat.0.jpg')])
    dataset = tf.contrib.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(_parse).repeat(20)
    iterator = dataset.make_one_shot_iterator()
    sess = tf.Session()
    try:
        cnt = 0
        while True:
            imsave(os.path.join(save_path, "%d.jpg" % cnt),
                   sess.run(iterator.get_next()))
            cnt += 1
    except tf.errors.OutOfRangeError:
        print('End of dataset iterator.')
    sess.close()


if __name__ == '__main__':
    main()
