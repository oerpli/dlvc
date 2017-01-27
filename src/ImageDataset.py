#!/usr/bin/python3
import pickle as pickle
import os
import numpy as np
from ClassificationDataset import ClassificationDataset

class ImageDataset(ClassificationDataset):
    # A dataset, consisting of multiple samples/images
    # and corresponding class labels.

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


    # To save images to file/display them.  Maybe comment this out because
    # libraries not supported on server
    def save(self,name,sid):
        from PIL import Image
        img = Image.fromarray(self.data[0], 'RGB')
        img.save("imgs/{0}_{1}.png".format(name,sid))

    def show(self,sid):
        from matplotlib import pyplot as plt
        plt.imshow(self.data[sid], interpolation='nearest')
        plt.show()