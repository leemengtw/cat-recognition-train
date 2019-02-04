import os
from math import ceil
from glob import glob
import random
import logging
import tensorflow as tf


MEAN = [v * 255 for v in [0.485, 0.456, 0.406]]
STD = [v * 255 for v in [0.229, 0.224, 0.225]]


class Dataset():

    def __init__(
            self,
            folder,
            train=True,
            size=(224, 224),
            batch_size=64,
            shuffle_buffer=None,
            epochs=1,
            val=False,
            valratio=0.1,
            random_seed=0):
        random.seed(random_seed)
        self.train = train
        self.size = size
        self.val = val
        self.root = os.path.join(folder, "train" if train else "test1")
        paths = sorted(glob(os.path.join(self.root, "*.jpg")))

        def _normalize(img):
            # imagenet mean and std
            mean = tf.constant([[MEAN]], dtype=tf.float32)
            std = tf.constant([[STD]], dtype=tf.float32)
            return (img - mean) / std

        if self.train:
            cat_cnt = sum("cat" in p for p in paths)
            dog_cnt = len(paths) - cat_cnt
            cat_vsize = int(cat_cnt * valratio)
            dog_vsize = int(dog_cnt * valratio)
            cat_paths = [p for p in paths if "cat" in p]
            dog_paths = [p for p in paths if "dog" in p]
            random.shuffle(cat_paths)
            random.shuffle(dog_paths)
            train_paths = cat_paths[:-cat_vsize] + dog_paths[:-dog_vsize]
            val_paths = sorted(cat_paths[-cat_vsize:] + dog_paths[-dog_vsize:])

            def to_dataset(cur_paths, is_train=True):
                cur_labels = [0 if "dog" in p else 1 for p in cur_paths]
                cur_labels = tf.data.Dataset.from_tensor_slices(tf.constant(cur_labels))
                cur_paths = tf.data.Dataset.from_tensor_slices(tf.constant(cur_paths))
                dataset = tf.data.Dataset.zip((cur_paths, cur_labels))
                if is_train:
                    dataset = dataset.apply(
                        tf.data.experimental.shuffle_and_repeat(
                            shuffle_buffer if shuffle_buffer else self.train_length,
                            epochs))

                    def _random_resize(img):
                        out_size = tf.random_uniform(shape=[2], minval=.8, maxval=1.)
                        out_size = tf.cast(
                                out_size * tf.cast(tf.shape(img)[:2], tf.float32), tf.int32)
                        return tf.image.random_crop(img, [out_size[0], out_size[1], 3])

                    augs = [
                        (tf.image.random_hue, (.1,)),
                        (tf.image.random_saturation, (.8, 1.2)),
                        (tf.image.random_contrast, (.3, 1.)),
                        (_random_resize, None)]

                def _process(img, label):
                    label = tf.cast(label, tf.float32)
                    img = tf.image.decode_jpeg(tf.read_file(img))
                    if is_train:
                        aug_prob = tf.random_uniform(shape=[4], minval=0., maxval=1.)
                        img = tf.image.random_flip_left_right(img)
                        for i, (func, args) in enumerate(augs):
                            img = tf.cond(
                                tf.math.greater(aug_prob[i], .5),
                                (lambda: func(img, *args)) if args else (lambda: func(img)),
                                lambda: img)
                    return _normalize(tf.image.resize_images(img, self.size)), label

                return dataset.map(_process)

            self.train_length = len(train_paths)
            self.val_length = len(val_paths)
            self.val_dataset = to_dataset(val_paths, False).batch(batch_size)
            self.train_dataset = to_dataset(train_paths, True).batch(batch_size)
            self.dataset = self.val_dataset if val else self.train_dataset
            self.train_total_batches = ceil(self.train_length * epochs / batch_size)
            self.val_total_batches = ceil(self.val_length / batch_size)
            self.train_batch_per_epoch = ceil(self.train_total_batches / epochs)
            self.val_batch_per_epoch = self.val_total_batches
            self.train_iterator = self.train_dataset.make_one_shot_iterator()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
        else:
            self.length = len(paths)
            self.total_batches = ceil(self.length / batch_size)
            self.batch_per_epoch = self.total_batches
            paths = tf.constant(paths)
            dataset = tf.data.Dataset.from_tensor_slices(paths)
            self.dataset = dataset.map(
                lambda img: _normalize(tf.image.resize_images(
                    tf.image.decode_jpeg(tf.read_file(img)), self.size))).batch(batch_size)
            self.iterator = self.dataset.make_initializable_iterator()

    def get_batch(self):
        if self.train:
            return self.val_iterator.get_next() if self.val else self.train_iterator.get_next()
        else:
            return self.iterator.get_next()

    def initialize(self, sess):
        sess.run(self.val_iterator.initializer if self.train else self.iterator.initializer)

    def get_total_batches(self):
        if self.train:
            return self.val_total_batches if self.val else self.train_total_batches
        else:
            return self.total_batches

    def get_batch_per_epoch(self):
        if self.train:
            return self.val_batch_per_epoch if self.val else self.train_batch_per_epoch
        else:
            return self.batch_per_epoch

    def eval(self):
        if not self.train:
            logging.warning("Dataset.eval takes no effect in test mode")
        self.val = True

    def train(self):
        if not self.train:
            logging.warning("Dataset.train takes no effect in test mode")
        self.val = False


def _test():
    epochs = 2
    is_train = True
    is_val = False
    d = Dataset("datasets", epochs=epochs, train=is_train, val=is_val)
    sess = tf.Session()
    if not is_train or is_val:
        d.initialize(sess)
    next_item = d.get_batch()
    i = 0
    b_per_epoch = d.get_batch_per_epoch()
    print(b_per_epoch, d.get_total_batches(), d.get_batch_per_epoch())
    import time
    t = time.time()
    for i in range(d.get_total_batches()):
        v, k = sess.run(next_item)
        print(i // b_per_epoch + 1, i % b_per_epoch + 1, end='\033[K\r')
    try:
        while True:
            _, _ = sess.run(next_item)
            assert False, \
                "Code here should not be runned; dataset not gone through"
    except tf.errors.OutOfRangeError:
        print("YAY\033[K")
    print("\n%f sec elapsed." % (time.time() - t))


if __name__ == "__main__":
    _test()
