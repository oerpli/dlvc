#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import l2
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
import numpy as np
import math

from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset

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
dir = '../../../datasets/cifar10/tinycifar10-hog/'
train = ImageVectorizer(HDF5FeatureVectorDataset(dir + 'features_tinycifar10_train.h5',cifar10_classnames))
val = ImageVectorizer(HDF5FeatureVectorDataset(dir + 'features_tinycifar10_val.h5',cifar10_classnames))
test = ImageVectorizer(HDF5FeatureVectorDataset(dir + 'features_tinycifar10_test.h5',cifar10_classnames))

floatCast = FloatCastTransformation()
offset = SubtractionTransformation.from_dataset_mean(train)
scale = DivisionTransformation.from_dataset_stddev(train)
transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCast)
transformationSequence.add_transformation(offset)
transformationSequence.add_transformation(scale)


print("Setting up preprocessing ...")
print("  Adding {}".format(type(floatCast).__name__))
print("  Adding {} [train] (value: {:02.2f})".format(type(offset).__name__,offset.value))
print("  Adding {} [train] (value: {:02.2f})".format(type(scale).__name__,scale.value))

print("Initializing minibatch generators ...")

train_batch = MiniBatchGenerator(train,64,transformationSequence)
val_batch = MiniBatchGenerator(val,100,transformationSequence)
test_batch = MiniBatchGenerator(test,100,transformationSequence)

train_batch.create()

print("  [train] {} samples, {} minibatches of size {}".format(train.size(), train_batch.nbatches(), train_batch.batchsize()))
print("  [val]   {} samples, {} minibatches of size {}".format(val.size(), val_batch.nbatches(), val_batch.batchsize()))

print("Initializing softmax classifier and optimizer ...")




bestAccuracyGlobal = 0.0
bestLearningRate = 0.0
bestWeightDecay = 0.0

fileNameModelGlobal = "model_best_global_softmax_ah.h5"
fileNameModel = "model_best_ah.h5"


for learningRatePow in range(0,7,1): #was 5,30,5
    learningRate = 1 / math.pow(2,learningRatePow)
    for weightDecayPow in range(0,7,1):
        weightDecay = 1 / math.pow(2,weightDecayPow)
        model = Sequential()
        model.add(Dense(W_regularizer = l2(weightDecay),output_dim=10, input_dim=144))
        model.add(Activation('softmax'))

        sgd = SGD(lr=learningRate, decay=0, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=["accuracy"])

        epochs = 200 # TODO CHANGE TO 200
        bestAccuracy = 0.0
        bestAccuracyAtEpoch = 0
        maxEpochWithoutImprovement = 20 # TODO CHANGE TO 20
        #print("Training for {} epochs ...".format(epochs))
        for epoch in range(0,epochs):
            loss = []
            acc_t = []
            acc_v = []
            train_batch.shuffle()
            for bid in range(0,train_batch.nbatches()):
                # train classifier
                b = train_batch.batch(bid)
                features = b[0]
                labels = to_categorical(b[1],10)
                metrics = model.train_on_batch(features, labels)
                # store loss and accuracy
                loss.append(metrics[0])
                acc_t.append(metrics[1])

            for bid in range(0,val_batch.nbatches()):
                b = val_batch.batch(bid)
                # test classifier
                features = b[0]
                labels = to_categorical(b[1],10)
                metrics = model.test_on_batch(features, labels)
                y = model.predict_on_batch(features)
                # store validation accuracy
                acc_v.append(metrics[1])

            # compute means over loss & accurracy
            m_loss = np.mean(loss)
            m_acc_t = np.mean(acc_t)
            m_acc_v = np.mean(acc_v)

            #print("[Epoch {:0>3}] loss: {:02.3f}, training accuracy: {:02.3f},
            #validation accuracy: {:02.3f}".format(epoch + 1,m_loss, m_acc_t,
            #m_acc_v))

            if m_acc_v > bestAccuracy:
                bestAccuracy = m_acc_v
                bestAccuracyAtEpoch = epoch
                #print("New best validation accuracy, saving model to
                #{}".format(fileNameModel))
                model.save("./" + fileNameModel)
                if bestAccuracy > bestAccuracyGlobal:
                    bestAccuracyGlobal = bestAccuracy
                    bestLearningRate = learningRate
                    bestWeightDecay = weightDecay
                    model.save("./" + fileNameModelGlobal)
            elif epoch - bestAccuracyAtEpoch > maxEpochWithoutImprovement:
                print("\r  learning rate={:01.5f}, weight decay={:01.5f}, accuracy: {:02.3f} (epoch {})".format(learningRate,weightDecay,bestAccuracy,bestAccuracyAtEpoch))
                break
           # print("\r  Epochs={:2.1%} no impr={:0>3.0%}, acc={:02.2f}, acc global={:02.2f}".format((epoch / epochs), ((epoch - bestAccuracyAtEpoch) / maxEpochWithoutImprovement), bestAccuracy, bestAccuracyGlobal),end="", flush=True)

print("")
print("Testing best model (learning rate={:01.5f}, weight decay={:01.5f}) on test set ...".format(bestLearningRate,bestWeightDecay))
print("  [test] {} samples, {} minibatches of size {}".format(test.size(), test_batch.nbatches(), test_batch.batchsize()))

model.load_weights("./" + fileNameModelGlobal)
test_batch.shuffle()
acc_test = []
m_acc_test = 0.0
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
print("  Accuracy: {:0.2%}".format(m_acc_test))

print("Done")