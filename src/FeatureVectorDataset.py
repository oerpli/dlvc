#!/usr/bin/python3
import numpy as np
from ImageDataset import ImageDataset

class FeatureVectorDataset:
    # A dataset, consisting of multiple feature vectors
    # and corresponding class labels.
    data = ImageDataset

    def size(self):
        return self.data.size()

    def nclasses(self):
        # Returns the number of different classes.
        # Class labels start with 0 and are consecutive.
        return self.data.nclasses()

    def classname(self, cid):
        # Returns the name of a class as a string.
        return self.data.classname(cid)

    def sample(self, sid):
        d = np.ndarray
        d = self.data.sample(sid)
        return (d[0].reshape((3 * 32 * 32)).astype(np.float64, copy=False),d[1])
        # Returns the sid-th sample in the dataset, and the
        # corresponding class label. Depending of your language,
        # this can be a Matlab struct, Python tuple or dict, etc.
        # Sample IDs start with 0 and are consecutive.
        # Throws an error if the sample does not exist.

