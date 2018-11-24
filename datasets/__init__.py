from datasets.mnist import get_mnist
from datasets.usps import get_usps
from datasets import mnistBig
from datasets import uspsBig

def get_loader(name, split, batch_size=50):
    if name == "mnist":
        return get_mnist(split,batch_size)

    if name == "usps":
        return get_usps(split,batch_size)

    if name == "mnistBig":
        return mnistBig.get_mnist(split,batch_size)


    if name == "uspsBig":
        return uspsBig.get_usps(split,batch_size)
