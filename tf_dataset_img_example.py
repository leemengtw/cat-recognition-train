import os
import tensorflow as tf
from scipy.misc import imsave


def main():

    def _aug(img):
        aug_val = tf.random_uniform(shape=[3], minval=.0, maxval=1.)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.cond(
                tf.less(aug_val[0], .5),
                lambda: tf.image.random_hue(img, .5),
                lambda: img)
        img = tf.cond(
                tf.less(aug_val[1], .5),
                lambda: tf.image.random_saturation(img, .5, 1.5),
                lambda: img)
        img = tf.cond(
                tf.less(aug_val[2], .5),
                lambda: tf.image.random_brightness(img, .5),
                lambda: img)
        return img

    def _parse(path):
        img = tf.image.decode_jpeg(tf.read_file(path))
        # tf.image.decode_png also works
        # both reads both jpeg png
        # then what's the difference wtf
        img = tf.cond(
                tf.less(tf.random_uniform(shape=[], minval=.0, maxval=1.), .6),
                lambda: _aug(img),
                lambda: img)
        img = tf.image.resize_images(img, (128, 128))
        return img
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, 'dataset_example_img_path')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    p = [os.path.join(
        root, 'datasets', 'train', 'cat.%d.jpg' % i) for i in range(2)]
    paths = tf.constant(p)
    dataset = tf.contrib.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(_parse).batch(2).repeat(16)
    iterator = dataset.make_one_shot_iterator()
    sess = tf.Session()
    e = 0
    try:
        while True:
            imgs = sess.run(iterator.get_next())
            imsave(os.path.join(save_path, "%d.jpg" % e), imgs[0])
            e += 1
            imsave(os.path.join(save_path, "%d.jpg" % e), imgs[1])
            e += 1
    except tf.errors.OutOfRangeError:
        print('Dataset ends.')
    sess.close()


if __name__ == '__main__':
    main()
