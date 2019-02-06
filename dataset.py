import os
from math import ceil
from glob import glob
import random
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
            valratio=0.1,
            random_seed=0):
        random.seed(random_seed)
        self.train = train
        self.size = size
        self.root = os.path.join(folder, "train" if train else "test1")
        paths = sorted(glob(os.path.join(self.root, "*.jpg")))
        self.val = False
        self.initialized = False

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
                    dataset = dataset.shuffle(
                        shuffle_buffer if shuffle_buffer else self.train_length)

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
                    # label = tf.cast(label, tf.float32)
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
            self.val_dataset = to_dataset(
                val_paths, False).batch(batch_size).prefetch(batch_size * 4)
            self.train_dataset = to_dataset(
                train_paths, True).batch(batch_size).prefetch(batch_size * 4)
            self.iterator = tf.data.Iterator.from_structure(
                self.val_dataset.output_types, self.val_dataset.output_shapes)
            self.train_initializer = self.iterator.make_initializer(self.train_dataset)
            self.val_initializer = self.iterator.make_initializer(self.val_dataset)
            self.train_nbatches = ceil(self.train_length / batch_size)
            self.val_nbatches = ceil(self.val_length / batch_size)
        else:
            self.length = len(paths)
            self.nbatches = ceil(self.length / batch_size)
            paths = tf.constant(paths)
            dataset = tf.data.Dataset.from_tensor_slices(paths)
            self.dataset = dataset.map(
                lambda img: _normalize(tf.image.resize_images(
                    tf.image.decode_jpeg(tf.read_file(img)), self.size))).batch(batch_size)
            self.iterator = self.dataset.make_initializable_iterator()
            self.initializer = self.iterator.initializer

    def initialize(self, sess, train=True):
        if train:
            sess.run(self.train_initializer)
            self.val = False
        else:
            sess.run(self.val_initializer if self.train else self.initializer)
            self.val = True
        self.initialized = True

    def __len__(self):
        if not self.initialized:
            raise Exception("Dataset not initialized!")
        if self.train:
            return self.val_nbatches if self.val else self.train_nbatches
        else:
            return self.nbatches

    def get_next(self):
        return self.iterator.get_next()


def _test():
    epochs = 2
    is_train = True
    is_val = False
    d = Dataset("datasets", train=is_train)
    sess = tf.Session()
    import time
    t = time.time()
    next_item = d.get_next()
    for e in range(1, epochs + 1):
        d.initialize(sess, not is_val)
        for i in range(1, len(d) + 1):
            v, k = sess.run(next_item)
            print(e, i, end='\033[K\r')
    try:
        while True:
            sess.run(next_item)
            assert False, \
                "Code here should not be runned; dataset not gone through"
    except tf.errors.OutOfRangeError:
        print("YAY\033[K")
    except Exception as exp:
        print(exp)
    print("\n%f sec elapsed." % (time.time() - t))


if __name__ == "__main__":
    _test()
