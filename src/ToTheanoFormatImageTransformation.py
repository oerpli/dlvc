
from IdentityTransformation import IdentityTransformation
from SampleTransformation import SampleTransformation
import numpy as np

class ToTheanoFormatImageTransformation(SampleTransformation):
    # Assumes input image with shape (rows,cols,channels) and returns image
    # with (channels, rows, cols)

    def __init__(self, shape):
        self.newshape = shape

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # Returns a 3D tensor wit hshape [c, rows,cols]
        return sample.reshape(self.newshape)