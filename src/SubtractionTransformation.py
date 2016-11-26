from SampleTransformation import SampleTransformation
from IdentityTransformation import IdentityTransformation

class SubtractionTransformation(SampleTransformation):
    # Subtract a scalar from all features.
    value = float

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation applied computation.
        if tform == None:
            tform = IdentityTransformation()
        meanSum = 0.0
        for i in range(0, dataset.size()):
            meanSum += tform.apply(dataset.sample(i)[0]).mean()
        mean = meanSum / dataset.size()
        return SubtractionTransformation(mean)

    def __init__(self, value):
        # Constructor.
        # value is a scalar to subtract.
        self.value = value

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return sample - self.value

    def value(self):
        # Return the subtracted value.
        return self.value