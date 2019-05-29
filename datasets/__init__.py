from datasets.mnist import get_mnist
from datasets.usps import get_usps
from datasets import mnistBig
from datasets import uspsBig
from datasets import coxs2v

def get_loader(name, split, batch_size=50):
    if name == "mnist":
        return get_mnist(split,batch_size)

    elif name == "usps":
        return get_usps(split,batch_size)

    elif name == "mnistBig":
        return mnistBig.get_mnist(split,batch_size)

    elif name == "uspsBig":
        return uspsBig.get_usps(split,batch_size)

    elif name == "coxs2v":
        return coxs2v.get_coxs2v(split, batch_size)

    else:
        raise Exception("Dataset name {} not supported".format(name))

