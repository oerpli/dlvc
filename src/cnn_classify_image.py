import keras as k

from PIL import Image

from ImageVectorizer import ImageVectorizer

import numpy as np

from PerChannelDivisionImageTransformation import PerChannelDivisionImageTransformation
from PerChannelSubtractionImageTransformation import PerChannelSubtractionImageTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
from ToTheanoFormatImageTransformation import ToTheanoFormatImageTransformation
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

pathModel = "../model_best.h5"
pathImage = "../specification/cat.jpg"
means = [125.31,122.91,113.8]
stds = [63.05,62.16,66.74]

print("  Model: {}".format(pathModel))
print("  Image: {}".format(pathImage))
print("  Means: {}".format(means))
print("  Stds: {}".format(stds))

print("Loading image ...")
img = Image.open(pathImage)
image = np.array(img)
print("  Input shape: {}".format(image.shape))

print("Loading classifier ...")
model = k.models.load_model(pathModel)
print("  Input shape: {}, {} classes".format(model.input_shape[1:],model.output_shape[1]))


def tfName(tf):
    return type(tf).__name__



resize = ResizeImageTransformation(model.input_shape[2])
floatCast = FloatCastTransformation()
offset = PerChannelSubtractionImageTransformation(means)
scale = PerChannelDivisionImageTransformation(stds)

shape = model.input_shape
shape = (1,) + shape[1:]
reshape = ToTheanoFormatImageTransformation(shape)
transformationSequence = TransformationSequence()
transformationSequence.add_transformation(resize)
transformationSequence.add_transformation(floatCast)
transformationSequence.add_transformation(offset)
transformationSequence.add_transformation(scale)
transformationSequence.add_transformation(reshape)


print("Preprocessing image ...")
print("  Transformations in order:")
print("    {}".format(tfName(resize)))
print("    {}".format(tfName(floatCast)))
print("    {} ({})".format(tfName(offset),offset.values))
print("    {} ({})".format(tfName(scale),scale.values))
print("    {}".format(tfName(reshape)))
image = transformationSequence.apply(image)
print("  Result: shape {}, dtype: {}, mean: {:0.3f}, std: {:0.3f}".format(image.shape, image.dtype, image.mean(), image.std()))

print("Classifying image ...")
y = model.predict(image,verbose = 1)
import operator
max_index, max_value = max(enumerate(y[0]), key=operator.itemgetter(1))
print("  Class scores: [{}]".format(" ".join(["{:0.2f}".format(i) for i in y[0]])))
print("  ID of most likely class: {} (score: {:0.3f})".format(max_index,max_value))