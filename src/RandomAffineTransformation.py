import random
import numpy as np
import scipy.ndimage as nd
from SampleTransformation import SampleTransformation
from math import cos,sin,pi

class RandomAffineTransformation(SampleTransformation):
    # Randomly crop samples to a given size.
    max_angle = float
    max_shear_x = float
    max_shear_y = float

    def __init__(self, max_angle, max_shear_x, max_shear_y):
        # Constructor.
        # Affine transformation is applied to images.
        self.max_angle = max_angle / 180 * pi # np.deg2rad(max_angle) #
        self.max_shear_x = max_shear_x
        self.max_shear_y = max_shear_y



    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor

        (x,y,c) = sample.shape

        angle = self.max_angle #* random.random()
        shear_x = self.max_shear_x * random.random()
        shear_y = self.max_shear_y * random.random()

        shear = np.identity(3)
        shear[1,0] = shear_x
        shear[0,1] = shear_y

        rot = np.identity(3)
        rot[0,0] = cos(angle)
        rot[1,1] = rot[0,0]
        rot[0,1] = sin(angle)
        rot[1,0] = -rot[0,1]
        #rot[2,0] = x - rot[0,0] * y * rot[0,1]
        #rot[2,1] = x - rot[1,0] * y * rot[1,1]
        xm = x - rot[0,0] * y * rot[0,1]
        ym = x - rot[1,0] * y * rot[1,1]

        center = np.identity(3)
        center[2,1] = 39 # -(y * sin(angle) * 0.5 + x * cos(angle) * 0.5)
        center[2,0] = 39 # -(y * cos(angle) * 0.5 - x * sin(angle) * 0.5)
        offset = [0,0,0]
        offset[0] = -39 #(y * sin(angle) * 0.5 + x * cos(angle) * 0.5)
        offset[1] = -39 #(y * cos(angle) * 0.5 - x * sin(angle) * 0.5)
        return nd.affine_transform(sample,rot,offset = offset)