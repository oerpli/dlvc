# Testing envoriment. not for actual use in project
import numpy as np
from TinyCifar10Dataset import TinyCifar10Dataset
from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from TransformationSequence import TransformationSequence


a = []
t1 = DivisionTransformation(1)
a.append(t1)

dataset = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",'train')
sample = dataset.sample(1)[0];
stdSum = 0.0
for i in range(0, dataset.size()):
    sample = dataset.sample(i)[0];    
    stdSum = stdSum + sample.std();

std = stdSum / dataset.size()

t2 = DivisionTransformation.from_dataset_stddev(dataset)
t3 = SubtractionTransformation.from_dataset_mean(dataset,t2)

ts = TransformationSequence()
ts.add_transformation(t2);
ts.add_transformation(t3);

sampleT = ts.apply(sample);
print('x')
