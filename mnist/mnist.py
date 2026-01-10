import ssl
ssl._create_default_https_context = ssl._create_unverified_context # this avoids ssl errors

import torch, torchvision, matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

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
    print(test_dataset[0])
    # visualize(test_dataset)

    train_dataset = torchvision.datasets.MNIST(root = 'mnist/data/', download = True, train=True, transform = ToTensor())
    test_dataset = torchvision.datasets.MNIST(root = 'mnist/data/', download = True, train=False, transform = ToTensor())
    print(len(train_dataset), len(test_dataset))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 100, shuffle = True)

    model=NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28, 28, device=device)

    X_numpy = X.squeeze().detach().cpu().numpy()
    plt.imshow(X_numpy, cmap='gray', vmin=0, vmax=1)
    plt.show()

    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

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
