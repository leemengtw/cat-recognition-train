"""
Dataloader, slightly modified by mnicnc404,
    referred from PyTorch (torch.utils.data)

PyTorch License:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
    Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio,
    Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
   America and IDIAP Research Institute nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
import glob
import os
# import multiprocessing as mp
import numpy as np
from scipy.misc import imread, imresize


class RandomSampler():

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class SequentialSampler():

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class BatchSampler():

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class CatDogSet():

    def __init__(self, dir_path, debug=False, size=(64, 64)):
        self.all_imgs = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))
        targets = []
        for p in dir_path:
            targets.append(0 if 'dog' in p else 1)
        # list indexing is super fast while accepting only 1 index.
        # here we want 1 index so no need to transfer to np array.
        self.length = len(targets)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Todo: data augmentation
        if self.debug:
            print(self.all_imgs[index], self.targets[index])
        return imresize(imread(self.all_imgs[index])), self.targets[index]


def cat_dot_collate():
    # Todo
    pass


class CatDogLoader():

    def __init__(
            self,
            dir_path,
            debug=False,
            size=(64, 64),
            batch_size=1,
            num_workers=2,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            drop_last=False):
        self.dataset = CatDogSet(dir_path, debug=debug, size=size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = cat_dot_collate

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(self.dataset)
                else:
                    sampler = SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
