from IdentityTransformation import IdentityTransformation
from SampleTransformation import SampleTransformation
import numpy as np

class PerChannelDivisionImageTransformation(SampleTransformation):
    # Perform per-channel division of of image samples with a scalar.
    values = []


    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.
        if(tform == None):
            tform = IdentityTransformation()

        channelPos = len(dataset.data.shape)-1
        channelCount = dataset.data.shape[channelPos]

        samples = dataset.sample(0)[0]
        datasetSize = dataset.size()
#        s = (datasetSize,) + dataset.sample(0)[0].shape[0:(channelPos-1)]
        s = (datasetSize,) + dataset.sample(0)[0].shape
        samples = np.resize(samples,s)
        samples = np.zeros_like(samples)
        stds = []
        for i in range(0, datasetSize):
            sample = dataset.sample(i)[0];
            for channel in range(0, channelCount):
                samples[i,...,channel] = sample[...,channel]


        for channel in range(0, channelCount):
            stds.append(samples[...,channel].std())

        return PerChannelDivisionImageTransformation(stds)

    def __init__(self, values):
        # Constructor.
        # values is a vector of c divisors, one per channel.
        # c can be any value > 0.
        self.values = values;


    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return sample / self.values

    def values(self):
        # Return the divisors.
        return self.values