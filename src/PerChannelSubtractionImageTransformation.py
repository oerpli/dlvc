from IdentityTransformation import IdentityTransformation
from SampleTransformation import SampleTransformation

class PerChannelSubtractionImageTransformation(SampleTransformation):
    # Perform per-channel subtraction of of image samples with a scalar.
    values = []
    channelCount = int

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset, independently for every color channel.
        # tform is an optional SampleTransformation to apply before computation.
        # samples must be 3D tensors with shape [rows,cols,channels].
        # rows, cols, channels can be arbitrary values > 0.
        if tform == None:
            tform = IdentityTransformation()
        meanSums = []
        
        channelPos = len(dataset.data.shape)-1
        channelCount = dataset.data.shape[channelPos]

        for channel in range(0, channelCount):
            meanSums.append(0);

        for i in range(0, dataset.size()):
            sample =  tform.apply(dataset.sample(i)[0])
            for channel in range(0, channelCount):
                meanSums[channel] += sample[...,channel].mean()

        for channel in range(0, channelCount):
            meanSums[channel] = meanSums[channel] / dataset.size()
       

        return PerChannelSubtractionImageTransformation(meanSums)

    def __init__(self, values):
        # Constructor.
        # values is a vector of c values to subtract, one per channel.
        # c can be any value > 0.
        self.values = values;
        self.channels = len(values)

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # The sample datatype must be single-precision float.
        return sample - self.values      

    def values(self):
        # Return the subtracted values.
        self.values