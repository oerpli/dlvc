import cifar_data as cd
import FeatureVectorDataset as f
import KnnClassifier as k

t = cd.TinyCifar10Dataset("../data",'train')
trainingset = f.ImageVectorizer(t)
test = cd.TinyCifar10Dataset("../data",'test')
testset = f.ImageVectorizer(test)

# initalize image classifier
c = k.KnnClassifier(5,'l2')
c.train(trainingset) # set trainingset

s50 = testset.sample(50)  # get some sample

prediction = c.predict(s50[0]) # try to classify the sample

print('Predicted {0}, should be {1}'.format(prediction,s50[1]))


print() # only here to set breakpoint