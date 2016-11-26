import ClassificationDataset as cd
import SampleTransformation as st
from math import ceil
from random import randint
import numpy as np

class MiniBatchGenerator:
    # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.
    data =  cd.ClassificationDataset
    size = int
    transformation = st.SampleTransformation
    indices = list(range(0,1))


    def __init__(self, dataset, bs, tform=None):
        # Constructor.
        # dataset is a ClassificationDataset to wrap.
        self.data = dataset
        # bs is an integer specifying the minibatch size.
        self.size = bs
        # tform is an optional SampleTransformation.
        # If given, tform is applied to all samples returned in minibatches.
        self.transformation = tform
        self.indices = list(range(0,self.data.size()))


    def batchsize(self):
        return self.size
        # Return the number of samples per minibatch.
        # The size of the last batch might be smaller.

    def nbatches(self):
        return ceil(self.data.size() / self.size)
        # Return the number of minibatches.

    def shuffle(self):
        # Shuffle the dataset samples so that each
        # ends up at a random location in a random minibatch.
        for i in range(len(self.indices)-1,0,-1):
            j = randint(0,i)
            self.indices[i],self.indices[j] = self.indices[j], self.indices[i]


    def batch(self, bid):
        labels = []
        ids = []
        # add first element outside of loop to determine shape of tensor
        i = bid * self.batchsize()
        sample = self.data.sample(self.indices[i])
        samples = np.expand_dims(sample[0],axis = 0)
        labels.append(sample[1])
        ids.append(self.indices[i])
        # add remaning bs - 1 elements
        for i in range(bid * self.batchsize() +1 , (bid +1) * self.batchsize()):
            if i < len(self.indices):
                sample = self.data.sample(self.indices[i])
                samples = np.append(samples,np.expand_dims(sample[0],axis=0),axis= 0)
                labels.append(sample[1])
                ids.append(self.indices[i])
        return (samples,labels,ids)
        # Return the bid-th minibatch.
        # Batch IDs start with 0 and are consecutive.
        # Throws an error if the minibatch does not exist.