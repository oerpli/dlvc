from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator
import numpy as np

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


floatCast = FloatCastTransformation()
offset = SubtractionTransformation.from_dataset_mean(train)
scale = DivisionTransformation.from_dataset_stddev(train)
transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCast)
transformationSequence.add_transformation(offset)
transformationSequence.add_transformation(scale)


print("Setting up preprocessing ...")
print(" Adding {}".format(type(floatCast).__name__))
print(" Adding {} [train] (value: {:02.2f})".format(type(offset).__name__,offset.value))
print(" Adding {} [train] (value: {:02.2f})".format(type(scale).__name__,scale.value)) 

print("Initializing minibatch generators ...")

train_batch = MiniBatchGenerator(train,64,transformationSequence)
val_batch = MiniBatchGenerator(val,100,transformationSequence)

train_batch.create();

print(" [train] {} samples, {} minibatches of size {}".format(train.size(), train_batch.nbatches(), train_batch.batchsize()))
print(" [val]   {} samples, {} minibatches of size {}".format(val.size(), val_batch.nbatches(), val_batch.batchsize()))

print("Initializing softmax classifier and optimizer ...")

model = Sequential()
model.add(Dense(output_dim=10, input_dim=144))
model.add(Activation('softmax'))

fileNameModel = "model_best.h5"
sgd = SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=["accuracy"])

epochs = 200
bestAccuracy = 0.0
bestAccuracyAtEpoch = 0
maxEpochWithoutImprovement = 20
print("Training for {} epochs ...".format(epochs))
for epoch in range(0,epochs):
    loss = []
    acc_t = []
    acc_v = []
    for bid in range(0,train_batch.nbatches()):
        # train classifier
        b = train_batch.batch(bid)
        features = b[0];
        labels = to_categorical(b[1],10);
        metrics = model.train_on_batch(features, labels)
        # store loss and accuracy
        loss.append(metrics[0])
        acc_t.append(metrics[1])

    for bid in range(0,val_batch.nbatches()):
        b = val_batch.batch(bid)
        # test classifier
        features = b[0];
        labels = to_categorical(b[1],10);
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
        model.save("../"+fileNameModel);
    elif epoch - bestAccuracyAtEpoch > maxEpochWithoutImprovement: 
        print("Validation accuracy did not improve for {} epochs, stopping".format(maxEpochWithoutImprovement))
        break



print("Done")