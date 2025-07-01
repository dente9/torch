import torch
from torch.nn import Linear
import torchvision

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class mynn(torch.nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        self.linear1 = Linear(196608, 10)


    def forward(self, x):
        x = self.linear1(x)
        return x

work = mynn()

for data in dataloader:
    imgs, targets = data
    output =torch.flatten(imgs)
    output= work(output)
    print(output.shape)

    



