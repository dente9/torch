import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

DataLoader = torch.utils.data.DataLoader(dataset, batch_size=64)

class mynn(nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        x = self.conv1(x)
        return x

nnwork = mynn()

writer = SummaryWriter('./logs')
step=0
for data in DataLoader:
    imgs, targets = data
    output = nnwork(imgs)
    print(imgs.shape)
    print(output.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    #torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step+=1

writer.close()
