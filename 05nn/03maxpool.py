import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

#diation 空洞卷积
#default stride=kernel_size
#if input size not suffice stride,ceil_mode=True will select the remain part to maxpool
dataset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
# [0,1,2,3,1],
# [1,2,1,0,0],
# [5,2,3,1,1],
# [2,1,0,1,1],
# ],dtype=torch.float32)

# input=torch.reshape(input,(-1,1,5,5))

class mynn(torch.nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        self.maxpool1 = torch.nn.MaxPool2d( kernel_size=3, ceil_mode=True)
    def forward(self, x):
        x = self.maxpool1(x)
        return x
nnwork = mynn()
# output = nnwork(input)
# print(output)
writer=SummaryWriter('./logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = nnwork(imgs)
    writer.add_images("output", output, step)

    step+=1

writer.close()