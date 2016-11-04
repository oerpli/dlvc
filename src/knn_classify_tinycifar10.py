from ImageVectorizer import ImageVectorizer
from TinyCifar10Dataset import TinyCifar10Dataset
from GridSearch import GridSearch

# The actual script
ratio = 0.01 #Default is 0.1

trainData = TinyCifar10Dataset("../data",'train')
valData = TinyCifar10Dataset("../data",'val')
testData = TinyCifar10Dataset("../data",'test')
trainSet = ImageVectorizer(trainData)
valSet = ImageVectorizer(valData)
testSet = ImageVectorizer(testData)


k_range = range(1,40,4)
cmp_range = ['l2','l1']
GridSearch(k_range,cmp_range).gridSearch(trainSet, valSet, testSet)
print() # only here to set breakpoint
