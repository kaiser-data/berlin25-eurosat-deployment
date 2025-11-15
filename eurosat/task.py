"""eurosat: A Flower / PyTorch app."""

import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

warnings.filterwarnings(
    "ignore",
    message=r"The currently tested dataset are",
    category=UserWarning,
)

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.bn3 = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 96 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


eurosat_train = None  # Cache dataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition EuroSAT data using IID partitioning."""
    global eurosat_train

    # Load dataset once
    if eurosat_train is None:
        eurosat_train = load_dataset("tanganke/eurosat", split="train")

    # Simple IID partitioning: divide dataset into equal chunks
    total_samples = len(eurosat_train)
    samples_per_partition = total_samples // num_partitions
    start_idx = partition_id * samples_per_partition

    # Last partition gets remaining samples
    if partition_id == num_partitions - 1:
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_partition

    # Get partition slice
    partition = eurosat_train.select(range(start_idx, end_idx))

    # Split partition: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Apply transforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # Create dataloaders
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def create_run_dir(precision: str = "fp32") -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp and precision
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path includes precision
    save_path = Path.cwd() / f"outputs/{precision}/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    return run_dir, str(save_path)