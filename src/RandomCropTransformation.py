import random
import numpy as np
from SampleTransformation import SampleTransformation

class RandomCropTransformation(SampleTransformation):
    # Randomly crop samples to a given size.
    W = int
    H = int

    def __init__(self, width, height):
        # Constructor.
        # Images are cropped randomly to the specified width and height.
        self.W = width
        self.H = height


    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
        # If rows < height or cols < width, an error is raised.
        (rows,cols, x) = sample.shape
        if rows < self.H or cols < self.W:
                raise NameError("Can't apply crop, image too small")
        else:
            x = random.randint(0,rows - self.H)
            y = random.randint(0,cols - self.W)
            return sample[x:x + self.H,y:y + self.W,:]