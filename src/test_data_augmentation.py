from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from TransformationSequence import TransformationSequence
from ResizeImageTransformation import ResizeImageTransformation
from HorizontalMirroringTransformation import HorizontalMirroringTransformation
from RandomCropTransformation import RandomCropTransformation
from RandomAffineTransformation import RandomAffineTransformation
from VerticalMirroringTransformation import VerticalMirroringTransformation
from RandomGrayScaleTransformation import RandomGrayScaleTransformation

from PIL import Image

import numpy as np
import os

imageFileName = "cat.jpg"

print("Loading image ...")

crop = RandomCropTransformation(25,25,0.5, True)
resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(0.5)
flipv = VerticalMirroringTransformation(0.5)
affine = RandomAffineTransformation(15,10,10,0.2)
gray = RandomGrayScaleTransformation(0.2)
transformationSequence = TransformationSequence()
transformationSequence.add_transformation(resize)
#transformationSequence.add_transformation(flipv)
transformationSequence.add_transformation(gray)
transformationSequence.add_transformation(fliph)
transformationSequence.add_transformation(affine)
transformationSequence.add_transformation(crop)
transformationSequence.add_transformation(resize)

for i in range(0,20):
    img = Image.open(imageFileName)
    image = np.array(img)
    print("  Input shape: {}".format(image.shape))

    image = transformationSequence.apply(image)


    #from matplotlib import pyplot as plt
    #plt.imshow(np.copy(image).astype('uint8'), interpolation='nearest')
    #plt.show()

    #transformedImage = Image.frombuffer("RGBX", image.shape[0:2], image)
    im = Image.fromarray(np.copy(image).astype('uint8'))
    #im = Image.frombuffer("RGBX", image.shape[0:2], image)
    path = os.path.join(os.getcwd(), "augmented_samples" , str(i) + "_" +imageFileName)

    dir = os.path.join(os.getcwd(), "augmented_samples")
    if not os.path.exists(dir):
        os.makedirs(dir)
    print("  Saving to: {}".format(path))
    im.save(path, "JPEG")