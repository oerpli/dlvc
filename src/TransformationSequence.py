from SampleTransformation import SampleTransformation

class TransformationSequence(SampleTransformation):
    # Applies a sequence of transformations
    # in the order they were added via add_transformation().


    def __init__(self):
        self.transformations = []

    def add_transformation(self, transformation):
        # Add a transformation (type SampleTransformation) to the sequence.
        self.transformations.append(transformation)
        return

    def get_transformation(self, tid):
        # Return the id-th transformation added via add_transformation.
        # The first transformation added has ID 0.
        return self.transformations[tid]

    def apply(self, sample):
        # Apply the transformation and return the transformed version.
        for i in range(0, len(self.transformations)):
            sample = self.get_transformation(i).apply(sample)
        return sample

    def size(self):
        return len(self.transformations)
