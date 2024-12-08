#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import argparse
import numpy as np
import random
import os

SEED = 102
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Two convolutional layers
        # Input: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (N, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (N, 64, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # reduces to (N, 64, 14, 14)

        # Two fully connected layers
        # Flatten after pooling: 64*14*14 = 12544
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))  # (N, 64, 14, 14)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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
    train_loader = utils.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=0)
    test_loader = utils.DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=0)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.train:
        for epoch in range(10):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            print(f"Epoch [{epoch + 1}/10], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        torch.save(model.state_dict(), "./models/cnn_model.pth")

    if args.test:
        model.load_state_dict(torch.load("./models/cnn_model.pth", map_location=device))
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if args.test_5_runs:
        accuracies = []
        for run in range(1, 6):
            model.load_state_dict(torch.load("./models/cnn_model.pth", map_location=device))
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            test_loss, test_acc = test(model, test_loader, criterion, device)
            accuracies.append(test_acc)
            print(f"Run {run}: Test Accuracy = {test_acc:.4f}")
        print(f"Average Test Accuracy over 5 runs: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the CNN model')
    parser.add_argument('--test', action='store_true', help='Test the CNN model')
    parser.add_argument('--test_5_runs', action='store_true',
                        help='Train and test the CNN model 5 times and report average accuracy')
    args = parser.parse_args()
    main(args)
