from ImageVectorizer import ImageVectorizer
from TinyCifar10Dataset import TinyCifar10Dataset
from GridSearch import GridSearch

# The actual script
ratio = 0.01; #Default is 0.1


trainData = TinyCifar10Dataset("../data",'train')
valData = TinyCifar10Dataset("../data",'val')
testData = TinyCifar10Dataset("../data",'test')
trainSet = ImageVectorizer(trainData)
valSet = ImageVectorizer(valData)
testSet = ImageVectorizer(testData)


GridSearch(range(1,40,3), ['l1','l2']).gridSearch(trainSet, valSet, testSet)
print() # only here to set breakpoint
