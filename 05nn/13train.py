import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from model import *
from torch.utils.tensorboard import SummaryWriter

learn_rate = 1e-2

train_data = torchvision.datasets.CIFAR10(
    "../data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)

test_data = torchvision.datasets.CIFAR10(
    "../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"train_data_size: {train_data_size}, test_data_size: {test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


net = Net()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

total_train_steps = 0
total_test_steps = 0
epoch = 10

writer = SummaryWriter("../logs")
for i in range(epoch):
    print(f"Epoch {i + 1} start \n-------------------------------")
    for data in train_dataloader:
        imgs, targets = data
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_steps += 1
        if total_train_steps % 100 == 0:
            print(f"train step {total_train_steps}, loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_steps)

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = net(imgs)

            preds = outputs.argmax(dim=1)
            accuracy = torch.eq(preds, targets).sum().item() / targets.size(0)
            total_accuracy += accuracy
            print(f"bath_accuracy: {total_accuracy}")

            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

    print(f"test step {total_test_steps}, loss: {loss.item()}")
    print(f"test accuracy: {total_accuracy/test_data_size}")
    print(f"test total_loss: {total_test_loss}")
    writer.add_scalar(
        "test_accuracy", total_accuracy / test_data_size, total_test_steps
    )
    writer.add_scalar("test_loss", total_test_loss, total_test_steps)
    total_test_steps += 1
    torch.save(net, f"net_{i}.pth")
    # torch.save(net.state_dict(), f"net_{i}.pth")
    print(f"model saved")
writer.close()
