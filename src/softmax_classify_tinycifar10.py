from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from MiniBatchGenerator import MiniBatchGenerator


from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence


dataSetName = 'train'
train = ImageVectorizer(TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName))
dataSetName = 'val'
val = ImageVectorizer(TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName))



float = FloatCastTransformation()
scale = DivisionTransformation.from_dataset_stddev(train,float)
offset = SubtractionTransformation.from_dataset_mean(train,scale)


train_batch = MiniBatchGenerator(train,64,offset)
val_batch = MiniBatchGenerator(val,100,offset)

sequence = TransformationSequence()


model = Sequential()
model.add(Dense(output_dim=10, input_dim=3072))
model.add(Activation('softmax'))



print("Done")