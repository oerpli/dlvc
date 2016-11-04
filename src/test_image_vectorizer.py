from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
import KnnClassifier as k

t = TinyCifar10Dataset("../data",'train')
x = ImageVectorizer(t)


# Definition of output functions
def first(dataset):
    print('{0} samples'.format(dataset.size()))
    print('{0} classes, name of class #1: {1}'.format(dataset.nclasses(),dataset.classname(1)))

def second(dataset):
    sid = 499
    img = dataset.sample(sid)
    x = img[1]
    y = img[0]
    print('Sample #{0}: {1}, shape: {2}'.format(sid, dataset.classname(img[1]),img[0].shape))
    imgs = dataset.devectorize(img[0])
    print('Shape after devectorization: {0}'.format(imgs.shape))
#    from matplotlib import pyplot as plt
#    plt.imshow(imgs, interpolation='nearest')
#    plt.show()

first(x)
second(x)

print() # only here to set breakpoint