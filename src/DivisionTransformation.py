class DivisionTransformation(SampleTransformation):
    # Divide all features by a scalar.

    @staticmethod
    def from_dataset_stddev(dataset, tform=None):
        # Return a transformation that will divide by the global standard deviation
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.

    def __init__(self, value):
        # Constructor.
        # value is a scalar divisor != 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.

    def value(self):
        # Return the divisor.