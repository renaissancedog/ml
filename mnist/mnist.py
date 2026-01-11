import ssl
ssl._create_default_https_context = ssl._create_unverified_context # this avoids ssl errors

import torch, torchvision, matplotlib.pyplot as plt, numpy as np
import torch.nn as nn
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

BATCH_SIZE=100

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def main():
    test_dataset = torchvision.datasets.MNIST(root = 'mnist/data/', download = True, train=False)
    # visualize(test_dataset)

    train_dataset = torchvision.datasets.MNIST(root = 'mnist/data/', download = True, train=True, transform = ToTensor())
    test_dataset = torchvision.datasets.MNIST(root = 'mnist/data/', download = True, train=False, transform = ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    model=NeuralNetwork().to(device)
    model.load_state_dict(torch.load("mnist/model.pth", weights_only=True))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inspect_model(model)

    num_epochs=0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), "mnist/model.pth")
    print("saved model")

def inspect_model(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)
    for i, layer in enumerate(model.linear_relu_stack):
        if hasattr(layer, "weight"):
            print(f"Layer {i} weight shape:", layer.weight.shape)
            print(f"Layer {i} bias shape:", layer.bias.shape)

def train(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss=loss_fn(logits, y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 99:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            comparison=pred.argmax(1) == y
            correct += comparison.type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct/size):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(correct, size-correct)

def visualize(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title([label])
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
