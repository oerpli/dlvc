from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from TransformationSequence import TransformationSequence
from ResizeImageTransformation import ResizeImageTransformation
from HorizontalMirroringTransformation import HorizontalMirroringTransformation
from RandomCropTransformation import RandomCropTransformation
from RandomAffineTransformation import RandomAffineTransformation
from VerticalMirroringTransformation import VerticalMirroringTransformation
from PIL import Image

import numpy as np

imageFileName = "cat.jpg"

print("Loading image ...")

crop = RandomCropTransformation(10,10, True)
resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(0.5)
flipv = VerticalMirroringTransformation(0.5)
affine = RandomAffineTransformation(20,15,15)
transformationSequence = TransformationSequence()
#transformationSequence.add_transformation(flipv)
#transformationSequence.add_transformation(affine)
#transformationSequence.add_transformation(fliph)
transformationSequence.add_transformation(crop)
transformationSequence.add_transformation(resize)

for i in range(0,5):
    img = Image.open(imageFileName)
    image = np.array(img)
    print("  Input shape: {}".format(image.shape))





    image = transformationSequence.apply(image)

    #transformedImage = Image.frombuffer("RGBX", image.shape[0:2], image)

    #transformedImage.save("new_"+imageFileName, "JPEG")


    from matplotlib import pyplot as plt
    plt.imshow(np.copy(image).astype('uint8'), interpolation='nearest')
    plt.show()
