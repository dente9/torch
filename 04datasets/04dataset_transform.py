import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,transform=dataset_transform,download=True)

print(train_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img, label = train_set[i]
    writer.add_image("train_set", img, i)

writer.close()