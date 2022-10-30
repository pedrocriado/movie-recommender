import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ConvNet import ConvNet

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 128
    num_epochs = 15
    lr = 0.0005

    train_datasets = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    test_datasets = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=True)

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Beginning Training")
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        for img, target in train_loader:
            # Send data to gpu if possible, if not keep on cpu
            img = img.to(device=device)
            target = target.to(device=device)

            # Forward pass
            output = model(img)
            loss = criterion(output, target)

            # Get gradients and update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("Finished Training")

    print("Getting Train Accuracy")
    check_accuracy(train_loader, model, device)
    print()

    print("Getting Test Accuracy")
    check_accuracy(test_loader, model, device)

if __name__ == "__main__":
    main()