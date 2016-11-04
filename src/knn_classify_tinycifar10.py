from ImageVectorizer import ImageVectorizer
from TinyCifar10Dataset import TinyCifar10Dataset
from GridSearch import GridSearch

# The actual script
ratio = 0.1; #Default is 0.1


print('Testing raw image features ...');

trainData = TinyCifar10Dataset("../data",'train',ratio)
valData = TinyCifar10Dataset("../data",'val',ratio)
testData = TinyCifar10Dataset("../data",'test',ratio)
trainSet = ImageVectorizer(trainData)
valSet = ImageVectorizer(valData)
testSet = ImageVectorizer(testData)


GridSearch().gridSearch(trainSet, valSet, testSet);
print() # only here to set breakpoint
