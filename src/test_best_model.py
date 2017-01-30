import operator
import keras as k
from keras.utils.np_utils import to_categorical
#from ImageVectorizer import ImageVectorizer

from MiniBatchGenerator import MiniBatchGenerator
from Cifar10Dataset import Cifar10Dataset

import numpy as np

from PerChannelDivisionImageTransformation import PerChannelDivisionImageTransformation
from PerChannelSubtractionImageTransformation import PerChannelSubtractionImageTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
from HorizontalMirroringTransformation import HorizontalMirroringTransformation
from RandomCropTransformation import RandomCropTransformation
from RandomAffineTransformation import RandomAffineTransformation
from ResizeImageTransformation import ResizeImageTransformation
from SubtractionTransformation import SubtractionTransformation
from RGBtoYCCTransformation import RGBtoYCCTransformation
from VerticalMirroringTransformation import VerticalMirroringTransformation


fileNameModel = "model_cnn_best.h5"

#model.load_weights("./" + fileNameModel)
model = k.models.load_model("./" + fileNameModel)

#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png', show_shapes = True)


print("Loading test dataset ...")
dir = "../../../datasets/cifar10/cifar-10-batches-py"

dataSetName = 'train'
train = Cifar10Dataset(dir,dataSetName) # load training set as well to calculate mean and stddev.

dataSetName = 'test'
test = Cifar10Dataset(dir,dataSetName)

convRGBtoYCC = RGBtoYCCTransformation()

floatCast = FloatCastTransformation()
offset = PerChannelSubtractionImageTransformation.from_dataset_mean(train)
scale = PerChannelDivisionImageTransformation.from_dataset_stddev(train)
crop = RandomCropTransformation(25,25, 0.08, True)
resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(0.5)
#flipv = VerticalMirroringTransformation(0.5) # bad for performance
transformation = TransformationSequence()
#trainingTransformationSequence.add_transformation(flipv) # decreases
#performance
transformation.add_transformation(fliph)
transformation.add_transformation(crop)
#trainingTransformationSequence.add_transformation(convRGBtoYCC)
transformation.add_transformation(resize)
transformation.add_transformation(floatCast)
transformation.add_transformation(offset)
transformation.add_transformation(scale)

#testingTransformationSequence = TransformationSequence()
##testingTransformationSequence.add_transformation(convRGBtoYCC)
#testingTransformationSequence.add_transformation(floatCast)
#testingTransformationSequence.add_transformation(offset)
#testingTransformationSequence.add_transformation(scale)





print("Setting up preprocessing ...")
def tfName(tf):
    return type(tf).__name__
def niceList(floatlist):
    return ", ".join(["{:0.2f}".format(i) for i in floatlist])

print(" Adding {}".format(tfName(floatCast)))
print(" Adding {} [train] (value:{})".format(tfName(offset),niceList(offset.values)))
print(" Adding {} [train] (value:{})".format(tfName(scale),niceList(scale.values)))
print("Initializing minibatch generators ...")


test_batch = MiniBatchGenerator(test,100,transformation)

#test_batch.create()

print("Testing model on test set ...")
print("  [test] {} samples, {} minibatches of size {}".format(test.size(), test_batch.nbatches(), test_batch.batchsize()))

acc_test = []
m_acc_test = 0.0

pred = [0] * 10
classcount = [0] * 10
wpred = [0] * 10

for bid in range(0,test_batch.nbatches()):
    b = test_batch.batch(bid)
    # test classifier
    features = b[0]
    labels = to_categorical(b[1],10)
    metrics = model.test_on_batch(features, labels)
    y = model.predict_on_batch(features)
    # store validation accuracy
    acc_test.append(metrics[1])
    # compute mean of accurracy
    m_acc_test = np.mean(acc_test)

    correct = b[1]
    for i in range(0,features.shape[0]):
        classcount[correct[i]] += 1
        max_index, max_value = max(enumerate(y[i]), key = operator.itemgetter(1))
        if correct[i] == max_index:
            pred[correct[i]] += 1
        else:
            wpred[max_index] += 1
acc = []
for i in range(0,10):
    acc.append(pred[i] / classcount[i])

print("  Accuracy: {:0.2%}".format(m_acc_test))
print("  Accuracy per class: {}".format(niceList(acc)))
print("  Misclassifications end up in class: {}".format(wpred))

print("Done")
