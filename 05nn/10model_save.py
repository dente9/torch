import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)
# save patter 1, save the whole model and parameters
torch.save(vgg16, "vgg16_method1.pth")

# save patter 2, save only the model parameters
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
