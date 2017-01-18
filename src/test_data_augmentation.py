from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from TransformationSequence import TransformationSequence
from ResizeImageTransformation import ResizeImageTransformation
from HorizontalMirroringTransformation import HorizontalMirroringTransformation
from RandomCropTransformation import RandomCropTransformation
from VerticalMirroringTransformation import VerticalMirroringTransformation
from PIL import Image

import numpy as np

imageFileName = "cat.jpg"

print("Loading image ...")
img = Image.open(imageFileName)
image = np.array(img)
print("  Input shape: {}".format(image.shape))

#resize = ResizeImageTransformation(32)
fliph = HorizontalMirroringTransformation(1)
flipv = VerticalMirroringTransformation(1)

crop = RandomCropTransformation(20,20)

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(flipv)
transformationSequence.add_transformation(fliph)
image = transformationSequence.apply(image)

#transformedImage = Image.frombuffer("RGBX", image.shape[0:2], image)

#transformedImage.save("new_"+imageFileName, "JPEG")


from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()
