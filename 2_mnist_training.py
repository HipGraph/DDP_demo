import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.distributed as dist

torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

# Model, loss function, optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5
losses = []

import time

if device.type == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()

# Training loop
for epoch in range(epochs):
    for data, target in train_loader:
        # Move data to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Zero the gradients
        output = model(data) # Forward pass
        loss = criterion(output, target) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        losses.append(loss.item()) # Store loss for plotting
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
print(f"Training time: {elapsed:.2f}s")

# Plot loss curve
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("MNIST Training Loss Curve")
# plt.show()
plt.savefig("./img/mnist_loss_curve.png")
