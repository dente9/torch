import torch
import torchvision

# load the model patter 1
vgg16 = torch.load("vgg16_method1.pth", weights_only=False)
print(vgg16)
# load the model patter 2, load only the parameters

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg116)
