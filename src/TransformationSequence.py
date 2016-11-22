class TransformationSequence(SampleTransformation):
    # Applies a sequence of transformations
    # in the order they were added via add_transformation().

    def add_transformation(self, transformation):
        # Add a transformation (type SampleTransformation) to the sequence.

    def get_transformation(self, tid):
        # Return the id-th transformation added via add_transformation.
        # The first transformation added has ID 0.

    def apply(self, sample):
        # Apply the transformation and return the transformed version.