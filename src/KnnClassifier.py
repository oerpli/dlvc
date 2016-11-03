#!/usr/bin/python3
import heapq as h
import numpy as np
import FeatureVectorDataset as f

class KnnClassifier:
    # k-nearest-neighbors classifier.
    K = int
    ns = str
    tdata = f.FeatureVectorDataset

    def __init__(self, k, cmp):
        # Ctor. k is the number of nearest neighbors to search for,
        # and cmp is a string specifying the distance measure to
        # use, namely `l1` (L1 distance) or `l2` (L2 distance).
        self.K = k
        if(cmp == 'l1'):
            self.ns = cmp
        elif(cmp =='l2'):
            self.ns = cmp
        else:
            print('{0} is not a valid norm. Using L2 instead'.format(cmp))
            self.ns = cmp

    def train(self, dataset):
        # Train on a dataset (type FeatureVectorDataset).
        self.tdata = dataset

    def predict(self, fvec):
        def normL1(fvec):
            return fvec.sum()

        def normL2(fvec):
            return (fvec * fvec).sum()

        norms = dict()
        norms['l1'] = normL1
        norms['l2'] = normL2


        heap = [] # heap with all distances etc
        for i in range(0,self.tdata.size()):
            s = self.tdata.sample(i)
            diff = norms[self.ns](fvec-s[0])
            #h.heappush(heap, (diff,s[1])) # don't use heappush but only append as heap property is not needed here. reduces runtime from nlogn to n
            heap.append((diff,s[1]))
        votes = []
        h.heapify(heap) # O(n) heap property
        for i in range(0,self.K): # remove K elements with smallest distance
            elem = h.heappop(heap) # remove element with lowest distance
            votes.append(elem[1]) # save class
        def most_common(lst):
            return max(set(lst), key=lst.count)
        return most_common(votes)
        # Return the predicted class label for a given feature vector fvec.
        # If the label is ambiguous, any of those in question is returned.

