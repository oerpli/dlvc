#!/usr/bin/python3
import heapq as h
import numpy as np
import FeatureVectorDataset as f
import math

class KnnClassifier:
    # k-nearest-neighbors classifier.
    K = int
    ns = str
    tdata = f.FeatureVectorDataset

    def __init__(self, k, cmp):
        # Ctor.  k is the number of nearest neighbors to search for,
        # and cmp is a string specifying the distance measure to
        # use, namely `l1` (L1 distance) or `l2` (L2 distance).
        self.K = k
        if(self.K < 1):
            print("Invalid k")
        if(cmp == 'l1'):
            self.ns = cmp
        elif(cmp == 'l2'):
            self.ns = cmp
        else:
            print('{0} is not a valid norm. Using L2 instead'.format(cmp))
            self.ns = cmp

    def train(self, dataset):
        # Train on a dataset (type FeatureVectorDataset).
        self.tdata = dataset

    def normL1(s,fvec):
        return np.absolute(fvec).sum()

    def normL2(s,fvec):
        return ((fvec * fvec)).sum()

    def most_common(s,lst):
        return max(set(lst), key=lst.count)

    def predict(self, fvec):
        heap = [] # heap with all distances etc
        for i in range(0,self.tdata.size()):
            s = self.tdata.sample(i)
            dist = (fvec - s[0])
            diff = 0.0
            if(self.ns == 'l1'):
                #diff2 = self.normL1(dist)
                diff = np.linalg.norm((dist), ord=1)
                #print(str(diff-diff2))
            elif(self.ns == 'l2'):
                #diff2 = math.sqrt(self.normL2(dist))
                diff = np.linalg.norm((dist), ord=2)
                #print(str(diff-diff2))
            #h.heappush(heap, (diff,s[1])) # don't use heappush but only append
            #as heap property is not needed here.  reduces runtime from nlogn
            #to n
            heap.append((diff,s[1]))
        votes = []
        h.heapify(heap) # O(n) heap property
        for i in range(0,self.K): # remove K elements with smallest distance
            elem = h.heappop(heap) # remove element with lowest distance
            votes.append(elem[1]) # save class
        return self.most_common(votes)
        # Return the predicted class label for a given feature vector fvec.
        # If the label is ambiguous, any of those in question is returned.

