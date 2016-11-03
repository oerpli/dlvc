from FeatureVectorDataset import FeatureVectorDataset

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