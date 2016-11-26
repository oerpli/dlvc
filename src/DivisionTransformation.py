from SampleTransformation import SampleTransformation

class DivisionTransformation(SampleTransformation):
    # Divide all features by a scalar.
    value = float

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.
        stdSum = 0.0
        for i in range(0, dataset.size()):
            sample = dataset.sample(i)[0];
            if (tform != None):
                sample = tform.apply(sample)
            stdSum = stdSum + sample.std();
        std = stdSum / dataset.size();
        return DivisionTransformation(std)

    def __init__(self, value):
        # Constructor.
        # value is a scalar divisor != 0.
        if(value == 0):
            print("Division by 0 prevented")
#            value = 1
        self.value = value

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.
        return sample / self.value

    def value(self):
        # Return the divisor.
        return self.value