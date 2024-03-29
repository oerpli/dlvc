from ImageVectorizer import ImageVectorizer
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset
from GridSearch import GridSearch

 # must match TinyCifar10Dataset
cifar10_classnames = {  0: 'airplane',
                        1: 'automobile',
                        2: 'bird',
                        3: 'cat',
                        4: 'deer',
                        5: 'dog',
                        6: 'frog',
                        7: 'horse',
                        8: 'ship',
                        9: 'truck'}

print('Testing HOG features ...')
dir = '../../../datasets/cifar10/tinycifar10-hog/'
trainSet = HDF5FeatureVectorDataset(dir + 'features_tinycifar10_train.h5',cifar10_classnames)
valSet = HDF5FeatureVectorDataset(dir + 'features_tinycifar10_val.h5',cifar10_classnames)
testSet = HDF5FeatureVectorDataset(dir + 'features_tinycifar10_test.h5',cifar10_classnames)

k_range = range(1,40,4)
cmp_range = ['l2','l1']
GridSearch(k_range,cmp_range).gridSearch(trainSet, valSet, testSet)
print() # only here to set breakpoint

