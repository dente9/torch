import torch
from torch.nn import ReLU
import torchvision
from torch.utils.tensorboard import SummaryWriter

input =torch.tensor([[1,-0.5],
                    [-1,3]])

input = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=False)
class mynn(torch.nn.Module):
    def __init__(self):
        super(mynn, self).__init__()
        self.relu = ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(x)
        return x

work = mynn()
writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs,targets = data
    output = work(imgs)

    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step+=1

writer.close()




