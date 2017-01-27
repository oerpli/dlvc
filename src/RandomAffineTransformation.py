import random
import numpy as np
import scipy.ndimage as nd
from SampleTransformation import SampleTransformation
from math import cos,sin,pi

import skimage as sk

from skimage import data
from skimage import transform

class RandomAffineTransformation(SampleTransformation):
    # Randomly crop samples to a given size.


    def __init__(self, max_angle, max_shear_x, max_shear_y, prob):
        # Constructor.
        # Affine transformation is applied to images.
        self.max_angle = np.deg2rad(max_angle)
        self.max_shear_x = np.deg2rad(max_shear_x)
        self.max_shear_y = np.deg2rad(max_shear_y)
        self.prob = prob

    # rotate
    def rot(self,angle,x,y):
        rot_pre = transform.SimilarityTransform(translation=[-x, -y])
        rot_post = transform.SimilarityTransform(translation=[x, y])
        tf_rotate = transform.SimilarityTransform(rotation= angle)
        return (rot_pre + (tf_rotate + rot_post))

    # shear in x direction
    def shearx(self,angle, x,y):
        pre = transform.SimilarityTransform(translation=[-y * sin(angle), -x * cos(angle)])
        post = transform.SimilarityTransform(translation=[y * sin(angle),  x * cos(angle)])
        shear = transform.AffineTransform(shear=angle)
        return (pre + (shear + post))

    # shear in y direction (rotate 90 + shear x + rotate back)
    def sheary(self,angle, x,y):
        pre = self.rot(np.deg2rad(90),x,y)
        post = self.rot(np.deg2rad(-90),x,y)
        shear = self.shearx(-angle, y,x) # y and x switched because rotated image
        return (pre + (shear + post))


    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor

        if random.random() < self.prob:
            # get random angles
            angle = self.max_angle * random.uniform(-1,1)
            shear_x = self.max_shear_x * random.uniform(-1,1)
            shear_y = self.max_shear_y * random.uniform(-1,1)

            # center of image
            y, x = np.array(sample.shape[:2]) / 2.

            # build transformation from parts
            tf = self.shearx(shear_x,x,y)
            tf += self.sheary(shear_y,x,y)
            tf += self.rot(angle,x,y)

            return transform.warp(sample,(tf).inverse, preserve_range = True)
        else:
            return sample