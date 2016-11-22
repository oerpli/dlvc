# Testing envoriment. not for actual use in project
import numpy as np
from TinyCifar10Dataset import TinyCifar10Dataset
from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation


#a = np.array('1 2 3 4')
#a = a-1
#print('[{0}]'.format(a))

dataset = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",'train')
sample = dataset.sample(1)[0];
stdSum = 0.0
for i in range(0, dataset.size()):
    sample = dataset.sample(i)[0];    
    stdSum = stdSum + sample.std();

std = stdSum / dataset.size()

sample2 = DivisionTransformation.from_dataset_stddev(dataset)
sample3 = SubtractionTransformation.from_dataset_mean(dataset,sample2)

print('x')
