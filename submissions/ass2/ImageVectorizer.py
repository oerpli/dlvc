from FeatureVectorDataset import FeatureVectorDataset
import numpy as np
from math import sqrt
class ImageVectorizer(FeatureVectorDataset):
    # Wraps an image dataset and exposes its contents.
    # Samples obtained using sample() are returned as 1D feature vectors.
    # Use devectorize() to convert a vector back to an image.

    def __init__(self, dataset):
        self.data = dataset
        # Ctor.  dataset is the dataset to wrap (type ImageDataset).

    def devectorize(self, fvec):
        x = round(sqrt(len(fvec) / 3))


        return fvec.astype('uint8', copy=False).reshape((x,x,3))
        # Convert a feature vector fvec obtained using sample()
        # back to an image and return the converted version.

    # implement the members of FeatureVectorDataset