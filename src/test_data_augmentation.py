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

crop = RandomCropTransformation(32,32)
resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(1)
flipv = VerticalMirroringTransformation(1)
affine = RandomAffineTransformation(10,0,0)
transformationSequence = TransformationSequence()
#transformationSequence.add_transformation(resize)
#transformationSequence.add_transformation(flipv)
transformationSequence.add_transformation(affine)
#transformationSequence.add_transformation(fliph)
#transformationSequence.add_transformation(crop)

for i in range(0,1):
    img = Image.open(imageFileName)
    image = np.array(img)
    print("  Input shape: {}".format(image.shape))





    image = transformationSequence.apply(image)

    #transformedImage = Image.frombuffer("RGBX", image.shape[0:2], image)

    #transformedImage.save("new_"+imageFileName, "JPEG")


    from matplotlib import pyplot as plt
    plt.imshow(np.copy(image).astype('uint8'), interpolation='nearest')
    plt.show()
