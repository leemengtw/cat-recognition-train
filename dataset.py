import os
from glob import glob
from tqdm import tqdm
from imageio import imread
import tensorflow as tf


class Dataset():

    def __init__(self, train=True, preload=True, size=(128, 128), batch_size=16):
        self.augs = [
            (tf.image.random_hue, (.1,)),
            (tf.image.random_saturation, (.8, 1.2)),
            (tf.image.random_contrast, (.3, 1.)),
            (self._random_resize, None)]
        self.train = train
        self.preload = preload
        self.size = size
        self.root = os.path.join("datasets", "train" if train else "test1")
        paths = sorted(glob(os.path.join(self.root, "*.jpg")))[:32]
        if preload:
            print("Loading images")
            self.imgs = [imread(p) for p in tqdm(paths)]
            dataset = tf.data.Dataset.from_generator(
                lambda: self.imgs, tf.int32, output_shapes=[None, None, 3])
        else:
            self.imgs = tf.constant(paths)
            dataset = tf.data.Dataset.from_tensor_slices(self.imgs)
        if self.train:
            labels = [0 if "dog" in p else 1 for p in paths]
            labels = tf.data.Dataset.from_tensor_slices(tf.constant(labels))
            dataset = tf.data.Dataset.zip((dataset, labels))
            dataset = dataset.shuffle(len(paths))
        self.dataset = dataset.map(self._process).batch(batch_size)
        self.length = len(paths)
        self.iterator = self.dataset.make_one_shot_iterator()

    @staticmethod
    def _random_resize(img):
        out_size = tf.random_uniform(shape=[2], minval=.8, maxval=1.)
        out_size = tf.cast(out_size * tf.cast(tf.shape(img)[:2], tf.float32), tf.int32)
        return tf.image.random_crop(img, [out_size[0], out_size[1], 3])

    def _aug(self, img):
        aug_prob = tf.random_uniform(shape=[4], minval=.0, maxval=1.)
        img = tf.image.random_flip_left_right(img)
        for i, (func, args) in enumerate(self.augs):
            img = tf.cond(
                tf.math.greater(aug_prob[i], .5),
                (lambda: func(img, *args)) if args else (lambda: func(img)),
                lambda: img)
        return img

    def _process(self, img, label=None):
        if not self.preload:
            img = tf.image.decode_jpeg(tf.read_file(img))
        if self.train:
            img = self._aug(img)
            return tf.image.resize_images(img, self.size), label
        else:
            return tf.image.resize_images(img, self.size)

    def get(self):
        return self.iterator.get_next()

    def reinitialize(self):
        self.iterator = self.dataset.make_one_shot_iterator()


def _test():
    # tf.enable_eager_execution()
    d = Dataset()
    sess = tf.Session()
    import numpy as np
    from imageio import imsave
    j = 0
    try:
        while True:
            a, b = sess.run(d.get())
            print(a.dtype)
            for i in range(a.shape[0]):
                imsave(os.path.join(
                    'test_dataset', '/%02d_%02d.jpg' % (i, j)), a.astype(np.uint8)[i])
            j += 1
    except tf.errors.OutOfRangeError:
        d.reinitialize()
    sess.close()


if __name__ == "__main__":
    _test()
