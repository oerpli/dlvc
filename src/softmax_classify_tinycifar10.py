from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
import numpy as np

from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence


dataSetName = 'train'
train = ImageVectorizer(TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName))
dataSetName = 'val'
val = ImageVectorizer(TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName))



floatCast = FloatCastTransformation()
offsetAndCast = SubtractionTransformation.from_dataset_mean(train,floatCast)
scaleAndOffsetAndCast = DivisionTransformation.from_dataset_stddev(train,offsetAndCast)

print("Setting up preprocessing ...")
print(" Adding {}".format(type(floatCast).__name__))
print(" Adding {} [train] (value: {:02.2f})".format(type(offsetAndCast).__name__,offsetAndCast.value))
print(" Adding {} [train] (value: {:02.2f})".format(type(scaleAndOffsetAndCast).__name__,scaleAndOffsetAndCast.value)) 

print("Initializing minibatch generators ...")

train_batch = MiniBatchGenerator(train,64,scaleAndOffsetAndCast)
val_batch = MiniBatchGenerator(val,100,scaleAndOffsetAndCast)

print(" [train] {} samples, {} minibatches of size {}".format(train.size(), train_batch.nbatches(), train_batch.batchsize()))
print(" [val]   {} samples, {} minibatches of size {}".format(val.size(), val_batch.nbatches(), val_batch.batchsize()))

print("Initializing softmax classifier and optimizer ...")

model = Sequential()
model.add(Dense(output_dim=10, input_dim=3072))
model.add(Activation('softmax'))


## TODO AB HIER

model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=["accuracy"])


epochs = 10
print("Training for {} epochs ...".format(epochs))
for epoch in range(0,epochs):
    loss = []
    acc_t = []
    acc_v = []
    for bid in range(0,train_batch.nbatches()):
        # train classifier
        b = train_batch.batch(bid)
        # store loss
        features = b[0];
        labels = to_categorical(b[1],10);
        _loss = model.train_on_batch(features, labels)
        loss.append(_loss)
        # store training accurracy ??? where to get this?
        acc_t.append(0)

    for bid in range(0,val_batch.nbatches()):
        b = val_batch.batch(bid)
        # test classifier
        model.test_on_batch(b[0], b[1])
        model.predict_on_batch(b[0])
        # store validation accuracy
        acc_v.append(0)
    # compute means over loss & accurracy
    m_loss = 0 #
    m_acc_t = 0 #
    m_acc_v = 0 #
    print("[Epoch {:0>3}] loss: {:02.3f}, training accuracy: {:02.3f}, validation accuracy: {:02.3f}".format(epoch + 1,m_loss, m_acc_t, m_acc_v))



print("Done")