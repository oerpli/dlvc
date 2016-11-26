from TinyCifar10Dataset import TinyCifar10Dataset
from ImageVectorizer import ImageVectorizer
import MiniBatchGenerator as mb




def minibatch_printer(m,n, shape = False) :
    b = m.batch(n)
    print("Minibatch #{} has {} samples".format(n,b[0].shape[0]))
    if shape:
        print(" Data shape: {}".format(b[0].shape))
    print(" First 10 sample IDs: {}".format(b[2][:10]))
    #print(" First 10 labels: {}".format(b[1][:10]))


def dataset_tester(dataset):
    print("=== Testing with {} ===".format(type(dataset).__name__))
    print()
    print("Dataset has {} samples".format(dataset.size()))
    m = mb.MiniBatchGenerator(dataset,60)
    print("Batch generator has {} minibatches, minibatch size: {}".format(m.nbatches(),m.batchsize()))
    print()

    minibatch_printer(m,0,True)
    minibatch_printer(m,66)
    print()
    print("Shuffling samples")
    print()
    m.shuffle()
    minibatch_printer(m,0)
    minibatch_printer(m,66)
    print()

dataSetName = 'train'
cifar = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName)
vector = ImageVectorizer(cifar)

dataset_tester(cifar)
dataset_tester(vector)

print("Done")