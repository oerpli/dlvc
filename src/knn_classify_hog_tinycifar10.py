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

print('Testing HOG features ...');
trainSet = HDF5FeatureVectorDataset('features_tinycifar10_train.h5',cifar10_classnames)
valSet = HDF5FeatureVectorDataset('features_tinycifar10_val.h5',cifar10_classnames)
testSet = HDF5FeatureVectorDataset('features_tinycifar10_test.h5',cifar10_classnames)

GridSearch().gridSearch(trainSet, valSet, testSet);
