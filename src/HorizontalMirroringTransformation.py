class HorizontalMirroringTransformation(SampleTransformation):
    # Perform horizontal mirroring of samples with a given probability.

    def __init__(self, proba):
        # Constructor.
        # proba is a value in [0,1] that determines how likely it is
        # that a sample is mirrored (as opposed to left unchanged).

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].