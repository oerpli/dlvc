import numpy as np
import skimage.transform  as t
import scipy as sp
from SampleTransformation import SampleTransformation

class ResizeImageTransformation(SampleTransformation):
    # Resize an image array to the needed size

    def __init__(self, size, keepRatio = False):
        self.size = size
        self.keepRatio = keepRatio

    def apply(self, sample):
        # Apply the transformation and return the transformed version.

        (rows,cols, x) = sample.shape

        if rows == cols and rows == self.size:
            return sample
        if (self.keepRatio):
            if rows < cols:
    #            if rows < self.smallerSize:
    #                raise NameError("Invalid image. Size too small")
                newsize = (self.size, int(cols * self.size / rows),x)
            else:
    #            if rows < self.smallerSize:
    #                raise NameError("Invalid image. Size too small")
                newsize = (int(rows * self.size / cols) ,self.size,x)
        else:
            newsize = (self.size,self.size)
#        print("  Input Image: shape {}, dtype: {}, mean: {:0.3f}, std: {:0.3f}".format(sample.shape, sample.dtype, sample.mean(), sample.std()))

        # first line is scikit-image, the other scipy
        # result = sp.misc.imresize(sample, newsize) # use this to use other resize library, (gives worse results)

        result = t.resize(sample,newsize, preserve_range=True) # Does not work on server
        #result = t.resize(sample,newsize) * 255 # Does the same, works on server
        return  result