import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter


# train_set = torchvision.datasets.CIFAR10(root='./data', train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0,drop_last=True)

img,target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step=0
for data in test_loader:
    imgs,targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_set", imgs, step)
    step+=1