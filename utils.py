import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize


def read_image_and_resize(path, size=(128, 128), debug=False):
    """
    Read a image file as a numpy.ndarray, resize it and return the resized
    images.
    """
    img = imread(path)

    img_resized = imresize(img, size)
    if debug:
        print('Image resized from {} to {}'.format(img.shape, img_resized.shape))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)

    return img_resized


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

    Parameters:
    -----------
    dir_path: relative path to image folder
    size: final image size after resize operation
    xname: .npy file of resized images if this function has been run before.
    xname: .npy file of labels if this function has been run before.

    Returns:
    --------
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
