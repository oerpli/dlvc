from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
from TransformationSequence import TransformationSequence
from ResizeImageTransformation import ResizeImageTransformation
from PIL import Image

import numpy as np

imageFileName = "cat.jpg"

print("Loading image ...")
img = Image.open(imageFileName)
image = np.array(img)
print("  Input shape: {}".format(image.shape))

resize = ResizeImageTransformation(32)

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(resize)
image = transformationSequence.apply(image)

transformedImage = Image.frombuffer("RGBX", image.shape[0:2], image)

transformedImage.save("new_"+imageFileName, "JPEG")
