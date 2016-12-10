import numpy as np
import skimage.transform  as t
import scipy as sp
from SampleTransformation import SampleTransformation

class ResizeImageTransformation(SampleTransformation):
    # Casts the sample datatype to single-precision float (e.g.
    # numpy.float32).
    smallerSize = int

    def __init__(self, size):
        self.smallerSize = size

    def apply(self, sample):
        # Apply the transformation and return the transformed version.

        (rows,cols, x) = sample.shape
        if rows < cols:
            if rows < self.smallerSize:
                raise NameError("Invalid image. Size too small")
            newsize = (self.smallerSize, int(cols * self.smallerSize / rows),x)
        else:
            if rows < self.smallerSize:
                raise NameError("Invalid image. Size too small")
            newsize = (int(rows * self.smallerSize / cols) ,self.smallerSize,x)

        # first line is scikit-image, the other scipy
        return t.resize(sample,newsize, preserve_range=True)
        #return sp.misc.imresize(sample, newsize) # use this to use other resize library