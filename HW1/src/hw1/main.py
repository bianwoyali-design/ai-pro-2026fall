import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch import nn, optim
from torchvision.transforms import v2


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, 2)
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output


def get_dataset():
    generator = torch.Generator()
    generator.manual_seed(42)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    BATCH_SIZE = 64

    trainset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        generator=generator,
    )

    testset = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def train(net, trainloader, criterion, optimizer, device, epochs):
    batch = 0

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if epoch == 0 and i == 0:
                print("Model device:", next(net.parameters()).device)
                print("Images device:", inputs.device)
                print("Labels device:", labels.device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"batch": batch, "loss": loss.item()})

            batch += 1

        epoch_loss = running_loss / len(trainloader)
        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})

        print(f"Epoch [{epoch + 1} / {epochs}] loss: {epoch_loss:.3f}")


def test(net, testloader, device, classes):
    net.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %"
    )
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this assignment.")

    device = torch.device("cuda")

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    momentums = [0.5, 0.9, 0.98]

    for momentum in momentums:
        print(f"\n===== momentum = {momentum} =====")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        trainloader, testloader = get_dataset()

        net = LeNet().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=momentum)

        wandb.init(
            project="aipro-hw1-cifar10",
            name=f"momentum={momentum}",
            config={
                "batch_size": 64,
                "learning_rate": 0.001,
                "momentum": momentum,
                "epochs": 10,
            },
        )

        wandb.define_metric("batch")
        wandb.define_metric("loss", step_metric="batch")

        wandb.define_metric("epoch")
        wandb.define_metric("epoch_loss", step_metric="epoch")

        train(net, trainloader, criterion, optimizer, device, epochs=10)

        test(net, testloader, device, classes)

        wandb.finish()


if __name__ == "__main__":
    main()
