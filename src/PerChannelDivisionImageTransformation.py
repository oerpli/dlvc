from IdentityTransformation import IdentityTransformation
from SampleTransformation import SampleTransformation

class PerChannelDivisionImageTransformation(SampleTransformation):
    # Perform per-channel division of of image samples with a scalar.
    value = []
    channels = int

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.
        if(tform == None):
            tform = IdentityTransformation()

        channelCount = dataset.data.shape[3]

        samples = dataset.sample(0)[0]
        datasetSize = dataset.size()
        s = (datasetSize,) + samples.shape
        samples = np.resize(samples,s)
        stds = [] 
        for channel in range(0, channelCount):
            channelSample = samples[...,channel]
            stds.append(samples.std())
        return PerChannelDivisionImageTransformation(std)

    def __init__(self, values):
        # Constructor.
        # values is a vector of c divisors, one per channel.
        # c can be any value > 0.
        self.values = values;
        self.channels = len(values)

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return sample / self.values

    def values(self):
        # Return the divisors.
        return self.values