from Cifar10Dataset import Cifar10Dataset

class TinyCifar10Dataset(Cifar10Dataset):
    def __init__(self, fdir, split, ratio = 0.1):
        super().__init__(fdir, split)
        self.split(ratio,True)