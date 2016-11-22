class MiniBatchGenerator:
    # Create minibatches of a given size from a dataset.
    # Preserves the original sample order unless shuffle() is used.

    def __init__(self, dataset, bs, tform=None):
        # Constructor.
        # dataset is a ClassificationDataset to wrap.
        # bs is an integer specifying the minibatch size.
        # tform is an optional SampleTransformation.
        # If given, tform is applied to all samples returned in minibatches.

    def batchsize(self):
        # Return the number of samples per minibatch.
        # The size of the last batch might be smaller.

    def nbatches(self):
        # Return the number of minibatches.

    def shuffle(self):
        # Shuffle the dataset samples so that each
        # ends up at a random location in a random minibatch.

    def batch(self, bid):
        # Return the bid-th minibatch.
        # Batch IDs start with 0 and are consecutive.
        # Throws an error if the minibatch does not exist.