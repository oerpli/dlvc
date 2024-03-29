from ImageVectorizer import ImageVectorizer
from TinyCifar10Dataset import TinyCifar10Dataset
from GridSearch import GridSearch

# The actual script
print('Testing raw image features ...')

trainData = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",'train')
valData = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",'val')
testData = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",'test')
trainSet = ImageVectorizer(trainData)
valSet = ImageVectorizer(valData)
testSet = ImageVectorizer(testData)


k_range = range(1,40,4)
cmp_range = ['l2','l1']
GridSearch(k_range,cmp_range).gridSearch(trainSet, valSet, testSet)
print() # only here to set breakpoint
