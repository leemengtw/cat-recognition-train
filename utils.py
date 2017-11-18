import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from imageio import imread
from skimage.transform import resize

from PIL import Image


def read_image_and_resize(file_path, size=(128, 128), debug=False):
    """
    Read a image file as a numpy.ndarray, return the resized images.

    Parameters
    ----------
    file_path: str
        relative file path of the image
    size: tuple
        final image size after resize operation

    Returns
    -------
    img_resized: numpy.ndarray of (size[0], size[1], #channels)
    """
    img = imread(file_path)

    # remove alpha channel for .png
    if img.shape[2] == 4:
        img = imread(get_rgb_image(file_path))

    # img_resized = imresize(img, size)
    img_resized = resize(img, size)
    if debug:
        import matplotlib.pyplot as plt
        print('Image resized from {} to {}'
              .format(img.shape, img_resized.shape))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)

    return img_resized


def load_image_dataset_to_pickle(
        dir_path='datasets/train/',
        xname='features.pkl',
        yname='targets.npy'):
    """
    If first run, read all the images in the specifed directory
    as X and their corresponding labels as y.
    `y = 0` indicates that it's a dog image, and `y = 1` indicates cat
    otherwise.
    Once this functions is run, a .npy file (for y) and
    a .pkl (for X: list of img arrays) will be saved in dir_path so
    the image reading and resizeing procedure won't be executed next time.

    Parameters
    ----------
    dir_path: relative path to image folder
    xname: .pkl file of resized images if this function has been run before.
    xname: .npy file of labels if this function has been run before.

    Returns
    -------
    X: ndarray of shape (#images, height, width, #channel)
    y: ndarray of shape (#images, label)
    """
    x_path = os.path.join(dir_path, xname)
    y_path = os.path.join(dir_path, yname)
    if os.path.exists(x_path) and os.path.exists(y_path):
        with open(x_path, 'rb') as f:
            X = pickle.load(f)
        return X, np.load(y_path)  # X is a list of different sized np arrays.
    X, y = [], []
    all_img_files = glob.glob(os.path.join(dir_path, '*.jpg'))
    for img_file in all_img_files:
        img = imread(img_file)
        label = 0 if 'dog' in img_file else 1
        X.append(img)
        y.append(label)
    with open(x_path, 'wb') as f:
        pickle.dump(X, f)
        print('list of imgs saved in %s' % x_path)
    y = np.array(y).reshape(-1, 1)
    np.save(y_path, y)
    return X, y


def load_image_dataset(
        dir_path='datasets/train/',
        xname='features.npy',
        yname='targets.npy',
        size=(300, 300)):
    """
    If first run, read and resize all the images in the specifed directory
    to the specified (height, width) as X and their corresponding labels as
    y. `y = 0` indicates that it's a dog image, and `y = 1` indicates cat
    otherwise.
    Once this functions is run, 2 .npy files will be saved in dir_path so
    the image reading and resizeing procedure won't be executed next time.

    Parameters
    ----------
    dir_path: relative path to image folder
    size: final image size after resize operation
    xname: .npy file of resized images if this function has been run before.
    xname: .npy file of labels if this function has been run before.

    Returns
    -------
    X: ndarray of shape (#images, height, width, #channel)
    y: ndarray of shape (#images, label)
    """
    x_path = os.path.join(dir_path, xname)
    y_path = os.path.join(dir_path, yname)
    if os.path.exists(x_path) and os.path.exists(y_path):
        return np.load(x_path), np.load(y_path)

    X, y = [], []
    all_img_files = glob.glob(os.path.join(dir_path, '*.jpg'))

    for img_file in all_img_files:
        img = read_image_and_resize(img_file, size=size)
        label = 0 if 'dog' in img_file else 1
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    np.save(x_path, X)
    np.save(y_path, y)
    return X, y


def show_images_horizontally(images, labels=[], lookup_label=None,
                             figsize=(15, 7)):
    """
    Show images in jupyter notebook horizontally w/ labels as title.

    Parameters
    ----------
    images: ndarray of shape (#images, height, width, #channels)
    labels: ndarray of shape (#images, label)
    lookup_label: dict
        indicate what text to render for every value in labels
        e.g. {0: 'dog', 1: 'cat'}

    """
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, imshow, axis

    fig = figure(figsize=figsize)
    for i in range(images.shape[0]):
        fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label:
            plt.title(lookup_label[labels[i][0]])
        imshow(images[i], cmap='Greys_r')
        axis('off')


def freeze_graph(
        out_path,
        out_node_name,
        graph=None,
        sess=None,
        meta_path=None,
        ckpt_path=None):
    """
    Freeze a trained model to a static graph file for needed node only.
    If graph is provided, sess should be provided too, vice versa, indicating
    that saving freshly trained model to a frozen graph. The passed session
    should be using the passed graph and be sure that the variables have been
    initialized.
    If meta_path is provided, ckpt_path should be provided too, vice versa,
    indicating that saving previously saved dynamic checkpoint to static graph.

    Parameters
    ----------
    out_path: path to save your frozen graph (including file name).
    out_node_name: name of needed ops node. Should be list or str.
    graph: tf graph to load for graph definition.
    sess: tf session to load for current variables status.
    meta_path: saved checkpoint meta graph for graph definition.
    """
    assert isinstance(out_node_name, str) or isinstance(out_node_name, list), \
        "out_node_name should be a str or a list of str"
    assert (graph is None) == (sess is None), \
        "graph and sess should be given simultaneously;\
         otherwise don't give either one."
    assert (meta_path is None) == (ckpt_path is None), \
        "meta_path and ckpt_path should be given simultaneously;\
         otherwise don't give either one."
    assert (graph is None) ^ (meta_path is None), \
        "Either use current graph or use meta data as def."

    if isinstance(out_node_name, str):
        out_node_name = [out_node_name]
    close_sess = False

    if meta_path is not None:
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        sess = tf.Session(graph=graph)
        saver = tf.train.import_meta_graph(meta_path, graph=graph)
        saver.restore(sess, ckpt_path)
        close_sess = True
    in_graph_def = graph.as_graph_def()
    out_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            in_graph_def,
            out_node_name)
    with tf.gfile.GFile(out_path, 'wb') as f:
        f.write(out_graph_def.SerializeToString())

    if close_sess:
        sess.close()


def get_rgb_image(file_path, delete_original_image=False):
    """
    Remove image's alpha channel and return the path of the new
    transformed jpg image

    Parameters
    ----------
    file_path: str
        relative path to the image with alpha channel
    delete_original_image: bool
        set to true to remove original image after transformation

    Returns
    -------
    transformed_file_path; str
        relative path to the image without alpha channel
    """
    png = Image.open(file_path)
    png.load()  # required for png.split()

    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

    transformed_file_path = file_path.replace('.png', '.jpg')
    background.save(transformed_file_path, 'JPEG', quality=80)

    # delete original image
    if delete_original_image:
        os.remove(file_path)

    return transformed_file_path
