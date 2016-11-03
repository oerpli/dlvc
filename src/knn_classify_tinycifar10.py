from TinyCifar10Dataset import TinyCifar10Dataset
import FeatureVectorDataset as f
import KnnClassifier as k
from ImageVectorizer import ImageVectorizer

t = TinyCifar10Dataset("../data",'train')
trainingset = ImageVectorizer(t)
test = TinyCifar10Dataset("../data",'test')
testset = ImageVectorizer(test)

# initalize image classifier
c = k.KnnClassifier(5,'l2')
c.train(trainingset) # set trainingset

s50 = testset.sample(50)  # get some sample

prediction = c.predict(s50[0]) # try to classify the sample

print('Predicted {0}, should be {1}'.format(prediction,s50[1]))


print() # only here to set breakpoint