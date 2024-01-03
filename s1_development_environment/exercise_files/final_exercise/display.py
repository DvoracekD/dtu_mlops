import torch
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'batch' is your batch of images with shape (batch_size, height, width)
batch = torch.load("../../../data/corruptmnist/train_images_0.pt")
batch_size, height, width = batch.shape
# batch = torch.rand((batch_size, height, width))

# Function to create a collage from random images in the batch
def create_collage(batch, num_rows, num_cols):
    # Number of images to display in the collage
    num_images = num_rows * num_cols
    
    # Randomly select images from the batch
    selected_indices = np.random.choice(batch_size, num_images, replace=True)
    selected_images = batch[selected_indices]

    # Create the collage
    collage = np.zeros((num_rows * height, num_cols * width))

    for i in range(num_rows):
        for j in range(num_cols):
            start_row, start_col = i * height, j * width
            collage[start_row:start_row + height, start_col:start_col + width] = selected_images[i * num_cols + j]

    return collage

# Set the number of rows and columns in the collage
num_rows, num_cols = 100, 100

# Create and display the collage
collage = create_collage(batch.numpy(), num_rows, num_cols)
plt.imshow(collage, cmap='gray')  # Use 'gray' colormap for single-channel images
plt.title('Random Image Collage')
plt.show()