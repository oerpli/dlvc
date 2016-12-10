import numpy as np
import skimage.transform  as t
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
            return t.resize(sample, (self.smallerSize, int(cols * self.smallerSize / rows),x), preserve_range=True)
        else:
            if rows < self.smallerSize:
                raise NameError("Invalid image. Size too small")
            return t.resize(sample, (int(rows * self.smallerSize / cols) ,self.smallerSize,x), preserve_range=True)