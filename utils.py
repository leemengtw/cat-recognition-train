import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize


def read_image_and_resize(path, size=(128, 128), verbose=False):
    """
    Read a image file as a numpy.ndarray, resize it and return the resized images.
    """
    img = imread(path)

    img_resized = imresize(img, size)
    if verbose:
        print('Image resized from {} to {}'.format(img.shape, img_resized.shape))
        plt.figure()
        plt.subplot(1, 2, 1);plt.imshow(img)
        plt.subplot(1, 2, 2);plt.imshow(img_resized)

    return img_resized


def load_image_dataset(dir_path='datasets/train/', dataset_size=None,
                       size=(300, 300)):
    """
    Resize all the images in the specifed directory to the specified
    (height, width) as X and their corresponding labels as y. Where
    `y = 0` indicate it's a dog image while `y = 1` indicate cat image.

    Parameters:
    -----------
    dir_path: relative path to image folder
    dataset_size: total number of images to be included in the result,
        useful when there are too many images in the folder
    size: final image size after resize operation

    Returns:
    --------
    X: ndarray of shape (#images, height, width, #channel)
    y: ndarray of shape (#images, label)
    """
    import os
    import numpy as np

    X, y = [list() for _ in range(2)]
    all_img_files = os.listdir(dir_path)

    # if dataset_size is not specified, resize all the images
    dataset_size = dataset_size if dataset_size else len(all_img_files)

    # random pick files in the folder
    img_files = np.random.choice(all_img_files, dataset_size)
    for img_file in img_files:
        img = read_image_and_resize(dir_path + img_file, size=size)
        label = 0 if 'dog' in img_file else 1
        X.append(img);
        y.append(label)

    return (np.array(X), np.array(y).reshape(-1, 1))


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
        a = fig.add_subplot(1, images.shape[0], i + 1)
        if lookup_label:
            plt.title(lookup_label[labels[i][0]])
            assert labels.any(), 'labels not available'
        imshow(images[i], cmap='Greys_r')
        axis('off')










