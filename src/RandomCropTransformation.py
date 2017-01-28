import random
import numpy as np
from SampleTransformation import SampleTransformation

class RandomCropTransformation(SampleTransformation):
    # Randomly crop samples to a given size.

    def __init__(self, width, height, prob, varyCrop = True):
        # Constructor.
        # Images are cropped randomly to the specified width and height.
        self.W = width
        self.H = height
        self.varyCrop = varyCrop # varying instead of fixed crop
        self.prob = prob;       #probability of applying crop
        if varyCrop and self.H != self.W:
                raise NameError("Can't apply vary crop, width and height values differ")

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
        # If rows < height or cols < width, an error is raised.
        (rows,cols, x) = sample.shape
        if rows < self.H or cols < self.W:
                raise NameError("Can't apply crop, image too small")
        else:
            if random.random() < self.prob:
                if (self.varyCrop):
                    sizeMax = min(rows,cols)
                    sizeX = random.randint(self.W, sizeMax);
                    sizeY = sizeX
                else:
                   sizeX = self.H
                   sizeY = self.W
                x = random.randint(0, rows - sizeX)
                y = random.randint(0, cols - sizeY)
                return sample[x:x + sizeX,y:y + sizeY,:]
            else:
                return sample
