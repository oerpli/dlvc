import numpy as np
import skimage.transform  as t
import scipy as sp
from SampleTransformation import SampleTransformation

class RGBtoYCCTransformation(SampleTransformation):
    # Converts from RGB image space to YCC color space

    def apply(self, sample):
        # Apply the transformation and return the transformed version.

        (rows,cols, x) = sample.shape
        _sample = np.copy(sample)
        for row in range(0, rows):
            for col in range(0, cols):
                 y,cb,cr = RGBtoYCCTransformation._ycc(sample[row,col,0],sample[row,col,1],sample[row,col,2])
                 _sample[row,col,0] = y
                 _sample[row,col,1] = cb
                 _sample[row,col,2] = cr
    
        return  _sample

    @staticmethod
    def apply(data): # in (0,255) range
        (ids,rows,cols, x) = data.shape
        for id in range(0, ids):
            for row in range(0, rows):
                for col in range(0, cols):
                     y,cb,cr = RGBtoYCCTransformation._ycc(data[id,row,col,0],data[id,row,col,1],data[id,row,col,2])
                     data[id,row,col,0] = y
                     data[id,row,col,1] = cb
                     data[id,row,col,2] = cr
    

    @staticmethod
    def _ycc(r, g, b): # in (0,255) range
        y = .299*r + .587*g + .114*b
        cb = 128 -.168736*r -.331364*g + .5*b
        cr = 128 +.5*r - .418688*g - .081312*b
        return y, cb, cr