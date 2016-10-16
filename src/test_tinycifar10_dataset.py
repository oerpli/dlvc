import cifar_data as d

set_names = ['train','val','test']
sets = dict()

for name in set_names:
    sets[name] = d.TinyCifar10Dataset("../data",name)




def first(sets,name):
    set = d.ImageDataset
    set = sets[name]
    print('[{0}] {1} classes, name of class #1: {2}'.format(name,set.nclasses(),set.classname(1)))

def second(sets,name):
    set = d.ImageDataset
    set = sets[name]
    c = dict
    c = set.classcount()
    print('[{0}] {1} samples'.format(name,set.size()))
    for x in c.keys():
        print('  Class #{0}: {1} samples'.format(x,c[x]))
    print()

def third(sets,name):
    set = d.ImageDataset
    set = sets[name]
    sid = 499
    print('[{0}] Sample #{1}: {2}'.format(name,sid,set.sampleclassname(sid)))

outputs = [first,second,third]

for fun in outputs:
    for name in set_names:
        fun(sets,name)
    print()

print() # only here to set a breakpoint