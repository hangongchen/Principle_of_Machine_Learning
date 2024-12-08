import torch
import torch.nn as nn
import torch.optim as optim
import utils
import argparse
import numpy as np
import random
import os

# Set random seeds for reproducibility
SEED = 102
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class FeedForwardNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super(FeedForwardNet, self).__init__()
        # Two hidden layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input: (N, 1, 28, 28) -> (N, 784)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data_, target in train_loader:
        data_, target = data_.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data_.size(0)
        _, pred = torch.max(output, 1)
        correct += pred.eq(target).sum().item()
        total += data_.size(0)
    return total_loss / total, correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data_, target in test_loader:
            data_, target = data_.to(device), target.to(device)
            output = model(data_)
            loss = criterion(output, target)
            total_loss += loss.item() * data_.size(0)
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()
            total += data_.size(0)
    return total_loss / total, correct / total


def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = './Data/'
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    # Create dataset instances
    train_dataset = utils.MNISTDataset(train_images_path, train_labels_path)
    test_dataset = utils.MNISTDataset(test_images_path, test_labels_path)

    # Create DataLoaders
    train_loader = utils.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = utils.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Example of iterating through the data
    # (Here we just print shape and label of the first batch)
    for imgs, labels in train_loader:
        print("Batch of images shape:", imgs.shape)  # [64, 1, 28, 28]
        print("Batch of labels:", labels.shape)  # [64]
        break

    # Initialize model, loss function, optimizer
    model = FeedForwardNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.train:
        # Train the model
        for epoch in range(10):  # 10 epochs
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        torch.save(model.state_dict(), "./models/feedforward_model.pth")

    if args.test:
        # Test the model
        model.load_state_dict(torch.load("./models/feedforward_model.pth", map_location=device))
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if args.test_5_runs:
        accuracies = []
        model.load_state_dict(torch.load("./models/feedforward_model.pth", map_location=device))

        for run in range(1, 6):
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(10):
                train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)

            test_loss, test_acc = test(model, test_loader, criterion, device)
            accuracies.append(test_acc)
            print(f"Run {run}: Test Accuracy = {test_acc:.4f}")

        print(f"Average Test Accuracy over 5 runs: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the feedforward model')
    parser.add_argument('--test', action='store_true', help='Test the feedforward model')
    parser.add_argument('--test_5_runs', action='store_true',
                        help='Train and test the model 5 times and report average accuracy')
    args = parser.parse_args()
    main(args)
