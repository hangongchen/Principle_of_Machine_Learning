import os
import struct
import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.transform = transform
        self.images, self.labels = self._load_data(images_path, labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # img is currently in shape [28, 28], convert to [1, 28, 28] if needed
        img = img.unsqueeze(0)  # (C=1, H=28, W=28)

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_data(self, images_path, labels_path):
        # Load labels
        with open(labels_path, 'rb') as lbpath:
            # Magic number for label file: 2049 (0x00000801)
            # First 4 bytes: magic number, next 4 bytes: number of labels
            magic, num = struct.unpack('>II', lbpath.read(8))
            if magic != 2049:
                raise ValueError(f'Invalid magic number for labels file: {magic}')
            labels = torch.frombuffer(lbpath.read(num), dtype=torch.uint8)

        # Load images
        with open(images_path, 'rb') as imgpath:
            # Magic number for image file: 2051 (0x00000803)
            # First 4 bytes: magic number, next 4 bytes: number of images
            # Next 4 bytes: number of rows, next 4 bytes: number of columns
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            if magic != 2051:
                raise ValueError(f'Invalid magic number for images file: {magic}')
            image_data = imgpath.read(num * rows * cols)
            images = torch.frombuffer(image_data, dtype=torch.uint8)
            images = images.view(num, rows, cols).float() / 255.0  # normalize to [0,1]

        return images, labels


if __name__ == "__main__":
    # Set paths to your MNIST files:
    data_dir = './Data/'
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    # Create dataset instances
    train_dataset = MNISTDataset(train_images_path, train_labels_path)
    test_dataset = MNISTDataset(test_images_path, test_labels_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Example of iterating through the data
    # (Here we just print shape and label of the first batch)
    for imgs, labels in train_loader:
        print("Batch of images shape:", imgs.shape)  # [64, 1, 28, 28]
        print("Batch of labels:", labels.shape)  # [64]
        break
