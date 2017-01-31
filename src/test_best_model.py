from keras.models import Sequential
import keras.layers as Lay
import keras as k
from keras.regularizers import l2
from keras.layers import Dropout, InputLayer, Activation
from keras.utils.np_utils import to_categorical
import operator

from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
from TinyCifar10Dataset import TinyCifar10Dataset
from Cifar10Dataset import Cifar10Dataset

import numpy as np
import random

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
from RandomGrayScaleTransformation import RandomGrayScaleTransformation

from VerticalMirroringTransformation import VerticalMirroringTransformation
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset

print("Loading Cifar10Dataset ...")
dir = "../../../datasets/cifar10/cifar-10-batches-py"

dataSetName = 'test'
test = Cifar10Dataset(dir,dataSetName)


#flipv = VerticalMirroringTransformation(0.5) # bad for performance
floatCast = FloatCastTransformation()
offset = PerChannelSubtractionImageTransformation.from_dataset_mean(test)
scale = PerChannelDivisionImageTransformation.from_dataset_stddev(test)
crop = RandomCropTransformation(28,28, 0.25, False)
resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(1)
affine = RandomAffineTransformation(15,10,10,0.2)
gray = RandomGrayScaleTransformation(0.2)


#testingTransformationSequence.add_transformation(fliph)
#testingTransformationSequence.add_transformation(crop)
#testingTransformationSequence.add_transformation(resize)
#testingTransformationSequence.add_transformation(affine)
#testingTransformationSequence.add_transformation(gray)
#testingTransformationSequence.add_transformation(floatCast)
#testingTransformationSequence.add_transformation(offset)
#testingTransformationSequence.add_transformation(scale)

def getBasicTransformationSequence():
    testingTransformationSequence = TransformationSequence()
    testingTransformationSequence.add_transformation(floatCast)
    testingTransformationSequence.add_transformation(offset)
    testingTransformationSequence.add_transformation(scale)
    return testingTransformationSequence 

testingTransformationSequences = []
testingTransformationSequence = getBasicTransformationSequence();

testingTransformationSequenceF = getBasicTransformationSequence();
testingTransformationSequenceF.add_transformation(fliph)

testingTransformationSequences.append(testingTransformationSequence);
testingTransformationSequences.append(testingTransformationSequenceF);

for i in range(0,3):
    testingTransformationSequenceC = getBasicTransformationSequence();
    testingTransformationSequenceC.add_transformation(crop)
    testingTransformationSequenceC.add_transformation(resize)
    testingTransformationSequences.append(testingTransformationSequenceC);

    testingTransformationSequenceA = getBasicTransformationSequence();
    testingTransformationSequenceA.add_transformation(affine)
    testingTransformationSequences.append(testingTransformationSequenceA);
    

print("")
print("Testing model on test set ...")

fileNameModel = "best_model.h5"

SIMULATE = False

if SIMULATE == False:
    model = k.models.load_model("./" + fileNameModel)

acc_test = []
m_acc_test = 0.0

pred = [0] * 10
classcount = [0] * 10
wpred = [0] * 10

batch_size = 100;
label_size = 10;
test_batches = []
for testingTransformationSequenceNo in range(0, len(testingTransformationSequences)):
    testingTransformationSequence = testingTransformationSequences[testingTransformationSequenceNo]
    test_batches.append(MiniBatchGenerator(test,100,testingTransformationSequence))

    
for bid in range(0,test_batches[0].nbatches()):
    ySum = np.zeros((batch_size, label_size))
    for generatorID in range(0, len(test_batches)):
        test_batch = test_batches[generatorID]
        b = test_batch.batch(bid)
        # test classifier
        features = b[0]
        labels = to_categorical(b[1],label_size)
        if SIMULATE:
            y= []
            for i in range (0, features.shape[0]):
                y.append([0,0,0,0,0,0,0,0,0,0])
                for j in range(0,label_size):
                    y[i][j] = random.random();
        else:
            metrics = model.test_on_batch(features, labels)
            y = model.predict_on_batch(features)
            #store validation accuracy
            acc_test.append(metrics[1])
            # compute mean of accurracy
            m_acc_test = np.mean(acc_test)
        ySum = np.matrix(y) + ySum # add up probalities

    correct = b[1]
    for i in range(0,features.shape[0]):
        classcount[correct[i]] += 1
        max_index, max_value = max(enumerate(ySum[i].T), key = operator.itemgetter(1))
        if correct[i] == max_index:
            pred[correct[i]] += 1
        else:
            wpred[max_index] += 1

acc = []
for i in range(0,10):
    acc.append(pred[i] / classcount[i])

accuracy = np.matrix(pred).sum() / test.size()
print("  Accuracy: {:0.2%}".format(m_acc_test))
#print("  Accuracy per class: {}".format(niceList(acc)))
print("  Misclassifications end up in class: {}".format(wpred))

print("Done")