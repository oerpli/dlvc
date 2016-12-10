import keras as k
import argparse

import sys

from PIL import Image

from ImageVectorizer import ImageVectorizer

import numpy as np

from PerChannelDivisionImageTransformation import PerChannelDivisionImageTransformation
from PerChannelSubtractionImageTransformation import PerChannelSubtractionImageTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
#from ToTheanoFormatImageTransformation import ToTheanoFormatImageTransformation
from HDF5FeatureVectorDataset import HDF5FeatureVectorDataset
from ResizeImageTransformation import ResizeImageTransformation



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


print("Parsing arguments ...")


parser = argparse.ArgumentParser(description='Classify given image after preprocessing with given model')
parser.add_argument('--means', dest='means', type=np.float32, nargs = 3,help='means per channel of the training set')
parser.add_argument('--stds', dest='stds', type=np.float32, nargs = 3 ,help='stds per channel of the training set')
parser.add_argument('--model', dest='model', help='file path of the model to be used for classification')
parser.add_argument('--image', dest='image', help='file path to the RGB image that should be classified')


defaultargs = '--means 125.31 122.91 113.8 --stds 63.05 62.16 66.74 --model model_cnn_best.h5 --image cat.jpg'.split(' ')
if len(sys.argv) > 1:
    args = parser.parse_args()
else:
    args = parser.parse_args(defaultargs)


def tfName(tf):
    return type(tf).__name__
def niceList(floatlist):
    return ", ".join(["{:0.2f}".format(i) for i in floatlist])



print("  Model: {}".format(args.model))
print("  Image: {}".format(args.image))
print("  Means: {}".format(niceList(args.means)))
print("  Stds:  {}".format(niceList(args.stds)))

print("Loading image ...")
img = Image.open(args.image)
image = np.array(img)
print("  Input shape: {}".format(image.shape))

print("Loading classifier ...")
model = k.models.load_model(args.model)
print("  Input shape: {}, {} classes".format(model.input_shape[1:],model.output_shape[1]))




resize = ResizeImageTransformation(model.input_shape[2])
floatCast = FloatCastTransformation()
offset = PerChannelSubtractionImageTransformation(args.means)
scale = PerChannelDivisionImageTransformation(args.stds)
#reshape = ToTheanoFormatImageTransformation() # not needed as we use tf now

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(resize)
transformationSequence.add_transformation(floatCast)
transformationSequence.add_transformation(offset)
transformationSequence.add_transformation(scale)

#transformationSequence.add_transformation(reshape)

print("Preprocessing image ...")
print("  Transformations in order:")
print("    {}".format(tfName(resize)))
print("    {}".format(tfName(floatCast)))
print("    {} ({})".format(tfName(offset),niceList(offset.values)))
print("    {} ({})".format(tfName(offset),niceList(scale.values)))
#print(" {}".format(tfName(reshape)))
image = transformationSequence.apply(image)
print("  Result: shape {}, dtype: {}, mean: {:0.3f}, std: {:0.3f}".format(image.shape, image.dtype, image.mean(), image.std()))

image = np.expand_dims(image,axis = 0)

print("Classifying image ...")
y = model.predict(image,verbose = 1,batch_size = 1)
import operator
max_index, max_value = max(enumerate(y[0]), key = operator.itemgetter(1))
print("  Class scores: [{}]".format(niceList(y[0])))
print("  ID of most likely class: {} (score: {:0.3f}) - {}".format(max_index,max_value, cifar10_classnames[max_index]))
