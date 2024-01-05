import torch
from torchvision import datasets, transforms

# Define a transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

dataset_path = "datasets"
# Download the MNIST dataset
mnist_train = datasets.MNIST(root=dataset_path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transform)

# Create DataLoader to access the data in batches
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

# # Optionally, you can save the datasets to a file using torch.save
torch.save(mnist_train, 'datasets/MNIST/processed/train_dataset.pt')
torch.save(mnist_test, 'datasets/MNIST/processed/test_dataset.pt')

# Optionally, you can also save the DataLoader objects
# torch.save(train_loader, 'datasets/MNIST/processed/train_loader.pt')
# torch.save(test_loader, 'datasets/MNIST/processed/test_loader.pt')