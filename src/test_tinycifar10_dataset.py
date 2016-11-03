from ImageDataset import ImageDataset
from TinyCifar10Dataset import TinyCifar10Dataset

def constructor(name):
   return TinyCifar10Dataset("../data",name)

# Definition of output functions
def first(sets,name):
    set =ImageDataset
    set = sets[name]
    print('[{0}] {1} classes, name of class #1: {2}'.format(name,set.nclasses(),set.classname(1)))

def second(sets,name):
    set = ImageDataset
    set = sets[name]
    c = dict
    c = set.classcount()
    print('[{0}] {1} samples'.format(name,set.size()))
    for x in c.keys():
        print('  Class #{0}: {1} samples'.format(x,c[x]))
    print()
    
def third(sets,name):
    set = ImageDataset
    set = sets[name]
    sid = 499
    print('[{0}] Sample #{1}: {2}'.format(name,sid,set.sampleclassname(sid)))


# Load data sets into dictionary and call all three output functions on each of the datasets

set_names = ['train','val','test']
sets = dict()
for name in set_names:
    sets[name] = constructor(name)

outputs = [first,second,third]

for fun in outputs:
    for name in set_names:
        fun(sets,name)
    print()

print() # only here to set breakpoint