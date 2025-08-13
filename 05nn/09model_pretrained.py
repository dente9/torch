import torchvision
import torch.nn as nn

vgg16_f = torchvision.models.vgg16(pretrained=False)
vgg16_t = torchvision.models.vgg16(pretrained=True)

vgg16_t.add_module("add_linear", nn.Linear(1000, 10))
# vgg16_t.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_t)

vgg16_f.classifier[6] = nn.Linear(4096, 10)
print(vgg16_f)
