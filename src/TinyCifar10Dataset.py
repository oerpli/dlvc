from Cifar10Dataset import Cifar10Dataset

class TinyCifar10Dataset(Cifar10Dataset):
    def __init__(self, fdir, split):
        super().__init__(fdir, split)
        self.split(0.1,True)