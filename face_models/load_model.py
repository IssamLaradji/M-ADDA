
import torch

from .inceptionresnetv2 import InceptionResNetV2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def load_model(model_arch,
               embedding_size=128,
               imgnet_pretrained=False):

    if model_arch == "resnet18":
        model = resnet18(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet34":
        model = resnet34(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet50":
        model = resnet50(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet101":
        model = resnet101(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "resnet152":
        model = resnet152(num_classes=embedding_size, pretrained=imgnet_pretrained)
    elif model_arch == "inceptionresnetv2":
        model = InceptionResNetV2(bottleneck_layer_size=embedding_size)
    else:
        raise Exception("Model architecture {} is not supported.".format(model_arch))

    return model