from FeatureVectorDataset import FeatureVectorDataset
import numpy as np
import h5py


class HDF5FeatureVectorDataset(FeatureVectorDataset):
    # A dataset stored in a HDF5 file including the datasets
    # features (n*f matrix) and labels (n-vector).

    def __init__(self, fpath, class_names):
        # Ctor.  fpath is a path to the HDF5 file.
        # class_names is a mapping from labels to
        # class names, for every label.

        with h5py.File('../datasets/' + fpath,'r') as hf:
            self.features = np.array(hf.get('features'))
            self.labels = np.array(hf.get('labels'))
        self.classNames = class_names


    def size(self):
        return self.features.shape[0]

    def nclasses(self):
        # Returns the number of different classes.
        # Class labels start with 0 and are consecutive.
        return self.classNames.size()

    def classname(self, cid):
        # Returns the name of a class as a string.
        return self.classNames[self.labels[cid]]

    def sample(self, sid):
        return (self.features[sid],self.labels[sid]);
