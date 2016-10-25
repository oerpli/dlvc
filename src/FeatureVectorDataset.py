#!/usr/bin/python3
import numpy as np
import cifar_data as cd

class FeatureVectorDataset:
    # A dataset, consisting of multiple feature vectors
    # and corresponding class labels.
    data = cd.ImageDataset

    def size(self):
        return self.data.size()

    def nclasses(self):
        # Returns the number of different classes.
        # Class labels start with 0 and are consecutive.
        return self.data.nclasses()

    def classname(self, cid):
        # Returns the name of a class as a string.
        return self.data.classname(cid);

    def sample(self, sid):
        d = np.ndarray
        d = self.data.sample(sid);
        return (d[0].reshape((3*32*32)),d[1])
        # Returns the sid-th sample in the dataset, and the
        # corresponding class label. Depending of your language,
        # this can be a Matlab struct, Python tuple or dict, etc.
        # Sample IDs start with 0 and are consecutive.
        # Throws an error if the sample does not exist.


class ImageVectorizer(FeatureVectorDataset):
    # Wraps an image dataset and exposes its contents.
    # Samples obtained using sample() are returned as 1D feature vectors.
    # Use devectorize() to convert a vector back to an image.

    def __init__(self, dataset):
        self.data = dataset
        # Ctor.  dataset is the dataset to wrap (type ImageDataset).

    def devectorize(self, fvec):
        return fvec.reshape((32,32,3))
        # Convert a feature vector fvec obtained using sample()
        # back to an image and return the converted version.

    # implement the members of FeatureVectorDataset
