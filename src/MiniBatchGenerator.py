import ClassificationDataset as cd
import SampleTransformation as st
from IdentityTransformation import IdentityTransformation
from math import ceil
from random import randint
import numpy as np

class MiniBatchGenerator:
    # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.
    data = cd.ClassificationDataset
    size = int
    transformation = st.SampleTransformation
    indices = list(range(0,1))
    samples = []
    labels = []
    ids = []


    def __init__(self, dataset, bs, tform=None):
        # Constructor.
        # dataset is a ClassificationDataset to wrap.
        self.data = dataset
        # bs is an integer specifying the minibatch size.
        self.size = bs
        # tform is an optional SampleTransformation.
        # If given, tform is applied to all samples returned in minibatches.
        self.transformation = tform
        if(tform == None):
            self.transformation = IdentityTransformation()
        self.indices = list(range(0,self.data.size()))
        self.create();

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
        for i in range(len(self.indices) - 1,0,-1):
            j = randint(0,i)
            self.indices[i],self.indices[j] = self.indices[j], self.indices[i]
        self.create(); # recreate batches

    # pre process batches
    def create(self):
        self.samples = []
        self.labels = []
        self.ids = []

        for i in range(0, self.nbatches()):
            sample, label, id = self._batch(i)
            self.samples.append(sample);
            self.labels.append(label);
            self.ids.append(id);
        return

    def batch(self, bid):
        return (self.samples[bid],self.labels[bid],self.ids[bid])

    # Crear
    def _batch(self, bid):
        if bid >= self.nbatches():
            print("Invalid batch id {}, only {} batches available".format(bid,self.nbatches()))
        labels = []
        ids = []

        samples = self.data.sample(0)[0]
        bs = self.batchsize() # default batch size
        if bid +1 == self.nbatches(): # fix size if it's the last batch
            bs = self.data.size() % self.batchsize()
        s = (bs,) + samples.shape
        samples = np.resize(samples,s)
        x = samples.shape
        # add remaning bs - 1 elements
        for i in range(bid * self.batchsize(), (bid + 1) * self.batchsize()):
            if i < len(self.indices):
                sample = self.data.sample(self.indices[i])
                vector = self.transformation.apply(sample[0])
                samples[i % self.batchsize(),...] = vector
                labels.append(sample[1])
                ids.append(self.indices[i])
        return (samples,labels,ids)
        # Return the bid-th minibatch.
        # Batch IDs start with 0 and are consecutive.
        # Throws an error if the minibatch does not exist.