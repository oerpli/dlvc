#!/usr/bin/python3
import _pickle as pickle
import os
import numpy as np
from pathlib import Path


class ImageDataset:
    # A dataset, consisting of multiple samples/images
    # and corresponding class labels.
    data = np.ndarray
    labels = list
    label_names = list

    def split(s, ratio, drop):
        indices = []
        unique, counts = np.unique(s.labels, return_counts=True)
        c_counts = dict(zip(unique, counts))
        for c in range(0,s.nclasses()):
            cutoff = c_counts[c] * ratio
            c_count = 0
            for i, x in enumerate(s.labels):
                if x == c :
                    c_count +=1
                    if np.logical_xor(drop,c_count > cutoff):
                        indices.append(i)

        indices.sort()
        s.data = s.data[indices]
        s.labels = s.labels[indices]

    def size(self):
        return len(self.data)
        # Returns the size of the dataset (number of images).

    def nclasses(self):
        return max(self.labels) + 1
        # Returns the number of different classes.
        # Class labels start with 0 and are consecutive.

    def classname(self, cid):
        return self.label_names[cid]
        # Returns the name of a class as a string.
    def sample(self, sid):
        return (self.data[sid], self.labels[sid])
        # Returns the sid-th sample in the dataset, and the
        # corresponding class label.  Depending of your language,
        # this can be a Matlab struct, Python tuple or dict, etc.
        # Sample IDs start with 0 and are consecutive.
        # The channel order of samples must be RGB.
        # Throws an error if the sample does not exist.

    def sampleclassname(self,sid):
        return self.classname(self.labels[sid])
        # Returns classname of a given sample
    def classcount(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


    # To save images to file/display them
    def save(self,name,sid):
        from PIL import Image
        img = Image.fromarray(self.data[0], 'RGB')
        img.save("imgs/{0}_{1}.png".format(name,sid))

    def show(self,sid):
        from matplotlib import pyplot as plt
        plt.imshow(self.data[sid], interpolation='nearest')
        plt.show()

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
class TinyCifar10Dataset(Cifar10Dataset):
    def __init__(self, fdir, split):
        super().__init__(fdir, split)
        self.split(0.1,True)