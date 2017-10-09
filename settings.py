import os

# trained model
META_PATH = 'models/model.ckpt.meta'
SAVE_PATH = 'models/model.ckpt'

# model meta
TRAIN_DIR = 'datasets/train/'
TRAIN_X_MEAN_NPY = os.path.join(TRAIN_DIR, 'train_x_mean.npy')
TRAIN_X_STD_NPY = os.path.join(TRAIN_DIR, 'train_x_std.npy')

