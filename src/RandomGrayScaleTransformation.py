import random
import numpy as np
from SampleTransformation import SampleTransformation

class RandomGrayScaleTransformation(SampleTransformation):
    # Randomly make images b/w

    def __init__(self, prob):
        # Constructor.
        self.prob = prob;       #probability of applying transformation

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        # sample must be a 3D tensor with shape [rows,cols,channels].
        # If rows < height or cols < width, an error is raised.
        if random.random() < self.prob:
            m = np.mean(sample,axis=2)
            sample[:,:,0] = m
            sample[:,:,1] = m
            sample[:,:,2] = m
            return sample
        else:
            return sample
