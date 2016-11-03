from FeatureVectorDataset import FeatureVectorDataset

class HDF5FeatureVectorDataset(FeatureVectorDataset):
    # A dataset stored in a HDF5 file including the datasets
    # features (n*f matrix) and labels (n-vector).

    def __init__(self, fpath, class_names):
        # Ctor. fpath is a path to the HDF5 file.
        # class_names is a mapping from labels to
        # class names, for every label.

    # implement the members of FeatureVectorDataset