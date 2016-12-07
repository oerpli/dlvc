from TinyCifar10Dataset import TinyCifar10Dataset
from ClassificationDataset import ClassificationDataset
from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence
from PerChannelDivisionImageTransformation import PerChannelDivisionImageTransformation
from PerChannelSubtractionImageTransformation import PerChannelSubtractionImageTransformation

# formatted output of a sample
def formatSample(sample) :
    shape = sample.shape;
    dtype = sample.dtype;
    mean = sample.mean();
    min = sample.min();
    max = sample.max();
    return "shape: {}, data type: {}, mean: {:02.1f}, min: {:02.1f}, max: {:02.1f}".format(shape, dtype, mean, min, max)

dataSetName = 'train'
dataSet = TinyCifar10Dataset("../../../datasets/cifar10/cifar-10-batches-py",dataSetName)

divisionTransformation = DivisionTransformation.from_dataset_stddev(dataSet)
subtractionTransformation = SubtractionTransformation.from_dataset_mean(dataSet)
perChannelDivisionImageTransformation = PerChannelDivisionImageTransformation.from_dataset_stddev(dataSet)
perChannelSubtractionImageTransformation = PerChannelSubtractionImageTransformation.from_dataset_mean(dataSet)
floatCastTransformation = FloatCastTransformation()
identityTransformation = IdentityTransformation()

print ("Computing SubtractionTransformation from TinyCifar10Dataset [{}] mean\n  Value: {:02.2f}".format(dataSetName, subtractionTransformation.value))
print ("Computing DivisionTransformation from TinyCifar10Dataset [{}] stdev\n  Value: {:02.2f}".format(dataSetName, divisionTransformation.value))
print ("Computing PerChannelSubtractionImageTransformation from TinyCifar10Dataset [{}] mean\n  Values: {:02.2f}, {:02.2f}, {:02.2f}".format(
dataSetName, perChannelSubtractionImageTransformation.values[0], perChannelSubtractionImageTransformation.values[1], perChannelSubtractionImageTransformation.values[2]))
print ("Computing PerChannelDivisionImageTransformation from TinyCifar10Dataset [{}] stdev\n  Values: {:02.2f}, {:02.2f}, {:02.2f}".format(
dataSetName, perChannelDivisionImageTransformation.values[0],  perChannelDivisionImageTransformation.values[1],  perChannelDivisionImageTransformation.values[2]))

sample = dataSet.sample(0)[0];
print ("First sample of TinyCifar10Dataset  [{}]: {}".format(dataSetName,formatSample(sample)))

transformedSample = IdentityTransformation().apply(sample);
print ("After applying IdentityTransformation: {}".format(formatSample(transformedSample)))

transformedSample = floatCastTransformation.apply(sample);
print ("After applying IdentityTransformation: {}".format(formatSample(transformedSample)))

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCastTransformation);
transformationSequence.add_transformation(subtractionTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> SubtractionTransformation: {}".format(formatSample(transformedSample)))

transformationSequence.add_transformation(divisionTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> SubtractionTransformation -> DivisionTransformation: {}".format(formatSample(transformedSample)))

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCastTransformation);
transformationSequence.add_transformation(perChannelSubtractionImageTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> PerChannelSubtractionImageTransformation: {}".format(formatSample(transformedSample)))

transformationSequence.add_transformation(perChannelDivisionImageTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> PerChannelSubtractionImageTransformation -> DivisionTransformation: {}".format(formatSample(transformedSample)))

