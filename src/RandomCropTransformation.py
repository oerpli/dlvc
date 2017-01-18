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
        return sample