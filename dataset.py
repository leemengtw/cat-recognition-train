import os
from math import ceil
from glob import glob
from psutil import virtual_memory
from tqdm import tqdm
from imageio import imread
import tensorflow as tf


class Dataset():

    def __init__(
            self,
            train=True,
            preload=True,
            size=(128, 128),
            batch_size=8,
            shuffle_buffer=None,
            epochs=2):
        if virtual_memory().total < 16 * 2**30 and preload:
            print("Not enough memory; Not preloading images into main memory.")
            preload = False
        self.augs = [
            (tf.image.random_hue, (.1,)),
            (tf.image.random_saturation, (.8, 1.2)),
            (tf.image.random_contrast, (.3, 1.)),
            (self._random_resize, None)]
        self.train = train
        self.preload = preload
        self.size = size
        self.root = os.path.join("datasets", "train" if train else "test1")
        paths = sorted(glob(os.path.join(self.root, "*.jpg")))[:35]
        self.length = len(paths)
        self.total_batches = ceil(self.length * epochs / batch_size)
        self.batch_per_epoch = ceil(self.total_batches / epochs)
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
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
                shuffle_buffer if shuffle_buffer else len(paths),
                epochs))
        self.dataset = dataset.map(self._process).batch(batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.get_batch_op = self.iterator.get_next()

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

    def get_batch(self):
        return self.get_batch_op

    def reinitialize(self):
        self.iterator = self.dataset.make_one_shot_iterator()

    def get_total_batches(self):
        return self.total_batches

    def get_batch_per_epoch(self):
        return self.batch_per_epoch


def _test():
    epochs = 2
    d = Dataset(epochs=epochs)
    sess = tf.Session()
    next_item = d.get_batch()
    i = 0
    b_per_epoch = d.get_batch_per_epoch()
    print(d.get_total_batches(), d.get_batch_per_epoch())
    for i in range(d.get_total_batches()):
        _ = sess.run(next_item)
        print(i // b_per_epoch + 1, i % b_per_epoch + 1)
    try:
        while True:
            _, _ = sess.run(next_item)
            print("WHAT")  # unexpected behavior
    except tf.errors.OutOfRangeError:
        print("YAY")
        d.reinitialize()


if __name__ == "__main__":
    _test()
