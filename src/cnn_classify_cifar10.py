from keras.models import Sequential
import keras.layers as Lay
import keras as k


from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
from TinyCifar10Dataset import TinyCifar10Dataset
from Cifar10Dataset import Cifar10Dataset

import numpy as np

from PerChannelDivisionImageTransformation import PerChannelDivisionImageTransformation
from PerChannelSubtractionImageTransformation import PerChannelSubtractionImageTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
from ToTheanoFormatImageTransformation import ToTheanoFormatImageTransformation
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


print("Loading Cifar10Dataset ...")
dir = "../../../datasets/cifar10/cifar-10-batches-py"
dataSetName = 'train'
train = Cifar10Dataset(dir,dataSetName)
dataSetName = 'val'
val = Cifar10Dataset(dir,dataSetName)
dataSetName = 'test'
test = Cifar10Dataset(dir,dataSetName)

floatCast = FloatCastTransformation()
offset = PerChannelSubtractionImageTransformation.from_dataset_mean(train)
scale = PerChannelDivisionImageTransformation.from_dataset_stddev(train)
reshape = ToTheanoFormatImageTransformation()
transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCast)
transformationSequence.add_transformation(offset)
transformationSequence.add_transformation(scale)
#transformationSequence.add_transformation(reshape)
print("Setting up preprocessing ...")
def tfName(tf):
    return type(tf).__name__
print(" Adding {}".format(tfName(floatCast)))
print(" Adding {} [train] (value:{})".format(tfName(offset)," ".join(["{:0.2f}".format(i) for i in offset.values])))
print(" Adding {} [train] (value:{})".format(tfName(scale)," ".join(["{:0.2f}".format(i) for i in scale.values])))
print("Initializing minibatch generators ...")

train_batch = MiniBatchGenerator(train,64,transformationSequence)
val_batch = MiniBatchGenerator(val,100,transformationSequence)
test_batch = MiniBatchGenerator(test,100,transformationSequence)

#train_batch.create()
#val_batch.create()
#test_batch.create()

print(" [train] {} samples, {} minibatches of size {}".format(train.size(), train_batch.nbatches(), train_batch.batchsize()))
print(" [val]   {} samples, {} minibatches of size {}".format(val.size(), val_batch.nbatches(), val_batch.batchsize()))
print()
print("Initializing CNN and optimizer ...")

model = Sequential()
model.add(Lay.Convolution2D(16,3,3,input_shape = (32,32,3),border_mode='same',activation='relu',dim_ordering="tf"))
model.add(Lay.MaxPooling2D((2,2),strides=(2,2),dim_ordering="tf"))
model.add(Lay.Convolution2D(32,3,3,border_mode='same',activation = 'relu',dim_ordering="tf"))
model.add(Lay.MaxPooling2D((2,2),strides=(2,2),dim_ordering="tf"))
model.add(Lay.Convolution2D(32,3,3,border_mode='same',activation = 'relu',dim_ordering="tf"))
model.add(Lay.MaxPooling2D((2,2),strides=(2,2),dim_ordering="tf"))
model.add(Lay.Flatten())
model.add(Lay.Dense(output_dim = 10,activation = 'softmax'))


weightDecay = 0.0001
learningRate = 0.001

sgd = SGD(lr=learningRate, decay=weightDecay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=["accuracy"])
model.summary()
print()

fileNameModel = "model_cnn_best.h5"

epochs = 100  #TODO set to 100
bestAccuracy = 0.0
bestAccuracyAtEpoch = 0
maxEpochWithoutImprovement = 10
print("Training for {} epochs ...".format(epochs))
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

    print("[Epoch {:0>3}] loss: {:02.3f}, training accuracy: {:02.3f}, validation accuracy: {:02.3f}".format(epoch + 1,m_loss, m_acc_t, m_acc_v))

    if m_acc_v > bestAccuracy:
        bestAccuracy = m_acc_v
        bestAccuracyAtEpoch = epoch
        print("New best validation accuracy, saving model to {}".format(fileNameModel))
        model.save(fileNameModel)
    elif epoch - bestAccuracyAtEpoch > maxEpochWithoutImprovement:
        print("Validation accuracy did not improve for {} epochs, stopping".format(maxEpochWithoutImprovement))
        print("Best validation accuracy: {:02.2f} (epoch {})".format(bestAccuracy,bestAccuracyAtEpoch))
        break

print("")
print("Testing model on test set ...")
print("  [test] {} samples, {} minibatches of size {}".format(test.size(), test_batch.nbatches(), test_batch.batchsize()))

#model.load_weights("./" + fileNameModel)
model = k.models.load_model("./" + fileNameModel)

test_batch.shuffle()
acc_test = []
m_acc_test = 0.0

pred = [0] * 10
classcount = [0] * 10

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
acc = []
for i in range(0,10):
    acc.append(pred[i]/classcount[i])
print("  Accuracy per class: {}".format(niceList(acc)))

print("Done")