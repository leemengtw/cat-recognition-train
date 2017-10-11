import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))


# trained model
META_PATH = '/'.join((PROJECT_PATH, 'models/model.ckpt.meta'))
SAVE_PATH = '/'.join((PROJECT_PATH, 'models/model.ckpt'))

# frozen model
PB_PATH = '/'.join((PROJECT_PATH, 'models/frozen.pb'))


# model meta
TRAIN_DIR = '/'.join((PROJECT_PATH, 'datasets/train/'))
TRAIN_X_MEAN_NPY = os.path.join(TRAIN_DIR, 'train_x_mean.npy')
TRAIN_X_STD_NPY = os.path.join(TRAIN_DIR, 'train_x_std.npy')
