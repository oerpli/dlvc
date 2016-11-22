class SubtractionTransformation(SampleTransformation):
    # Subtract a scalar from all features.

    @staticmethod
    def from_dataset_mean(dataset, tform=None):
        # Return a transformation that will subtract by the global mean
        # over all samples and features in a dataset.
        # tform is an optional SampleTransformation to apply before computation.

    def __init__(self, value):
        # Constructor.
        # value is a scalar to subtract.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # The sample datatype must be single-precision float.

    def value(self):
        # Return the subtracted value.