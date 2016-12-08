
from IdentityTransformation import IdentityTransformation
import theano.tensor as T
from SampleTransformation import SampleTransformation
import numpy as np

class ToTheanoFormatImageTransformation(SampleTransformation):
    # Assumes input image with shape (rows,cols,channels) and returns image
    # with (channels, rows, cols)

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,c].
        # Returns a 3D tensor wit hshape [c, rows,cols]
        transpose = (2,0,1)
        return np.transpose(sample,(2,0,1))