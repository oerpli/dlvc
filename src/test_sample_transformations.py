from TinyCifar10Dataset import TinyCifar10Dataset
from ClassificationDataset import ClassificationDataset
from SubtractionTransformation import SubtractionTransformation
from DivisionTransformation import DivisionTransformation
from IdentityTransformation import IdentityTransformation
from FloatCastTransformation import FloatCastTransformation
from TransformationSequence import TransformationSequence

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
floatCastTransformation = FloatCastTransformation()
identityTransformation = IdentityTransformation()

print ("Computing SubtractionTransformation from TinyCifar10Dataset [{}] mean  Value: {:02.2f}".format(dataSetName, subtractionTransformation.value))
print ("Computing DivisionTransformation from TinyCifar10Dataset [{}] stdev  Value: {:02.2f}".format(dataSetName, divisionTransformation.value))

sample = dataSet.sample(0)[0];
print ("First sample of TinyCifar10Dataset  [{}]: {}".format(dataSetName,formatSample(sample)))

identityTransformedSample = IdentityTransformation().apply(sample);
print ("After applying IdentityTransformation: {}".format(formatSample(identityTransformedSample)))

floatCastedSample = floatCastTransformation.apply(sample);
print ("After applying IdentityTransformation: {}".format(formatSample(floatCastedSample)))

transformationSequence = TransformationSequence()
transformationSequence.add_transformation(floatCastTransformation);
transformationSequence.add_transformation(subtractionTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> SubtractionTransformation: {}".format(formatSample(transformedSample)))

transformationSequence.add_transformation(divisionTransformation);
transformedSample = transformationSequence.apply(sample);
print ("After applying sequence FloatCast -> SubtractionTransformation -> DivisionTransformation: {}".format(formatSample(transformedSample)))

