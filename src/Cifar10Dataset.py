#!/usr/bin/python3
import _pickle as pickle
import os
import numpy as np
from ImageDataset import ImageDataset
from pathlib import Path


"""
CIFAR-10 Image classification dataset
Data available from and described at:
http://www.cs.toronto.edu/~kriz/cifar.html
If you use this dataset, please cite "Learning Multiple Layers of Features from
Tiny Images", Alex Krizhevsky, 2009.
http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
"""
class Cifar10Dataset(ImageDataset):
    # The CIFAR10 dataset.
    def __init__(self, fdir, split):
        # Ctor.  fdir is a path to a directory in which the CIFAR10
        # files reside (e.g.  data_batch_1 and test_batch).
        # split is a string that specifies which dataset split to load.
        # Can be 'train' (training set), 'val' (validation set) or 'test' (test
        # set).
        # TODO implement training/test/whatever split

        # local functions - no idea how private functions work
        def unpickle(file):
            fo = open(file, 'rb')
            dict = pickle.load(fo, encoding='latin1')
            fo.close()
            return dict
        def fixed(X):
            return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)

        if split == 'train' or split == 'val':
            files = [(fdir + '/data_batch_%i') % i for i in range(1,6)]
            x = 5
        elif split == 'test':
            files = [(fdir + '/test_batch')]
            x = 1

        if not Path(files[0]).is_file():
            dl_link = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            print("Data files not found.")
            abspath = os.path.abspath(fdir)
            print("Check if they are in the directory " + abspath)
            print("You can download them from: " + dl_link)
            raise NameError("Data not found")

        img_dim = 32
        img_channels = 3
        batch_size = 10000
        batches = 5


        img = np.zeros((batch_size * x, img_dim,img_dim, img_channels), dtype='uint8')
        labels = np.zeros(batch_size * x, dtype='int32')
        counter = 0
        batch = dict
        for i, file in enumerate(files):
            batch = unpickle(file)
            img[counter:counter + batch_size] = fixed(batch['data'])
            assert batch['data'].dtype == np.uint8
            labels[counter:counter + batch_size] = batch['labels']
            counter += batch_size
        self.data = img
        self.labels = labels

        meta = unpickle(fdir + '/batches.meta')
        self.label_names = meta['label_names']
        # filter if training/validation dataset
        if split == 'train':
            self.split(0.8,True) # get initial 80%
        elif split == 'val':
            self.split(0.8,False)# drop initial 80% (and take the other 20%)