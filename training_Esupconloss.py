import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

lambda=0.1

hierarchy = {
    0: {1: lambda},  # Control Nucleus ↔ Control Non-Nucleus (strong), Control Nucleus ↔ Stress Nucleus (weak)
    1: {0: lambda},
    2: {3: lambda},
    3: {2: lambda}
}


class EmpiricalSupConLoss(nn.Module):
    """Empirically Weighted Supervised Contrastive Loss."""
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, hierarchy=None):
        
        super(EmpiricalSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hierarchy = hierarchy or {}  # Default to empty dictionary if None is passed

    def forward(self, features, labels=None, mask=None):
        """
        Compute the empirical supervised contrastive loss.

        Args:
            features: Hidden vectors of shape [batch_size, n_views, ...].
            labels: Ground truth labels of shape [batch_size].
            mask: Contrastive mask of shape [batch_size, batch_size].
        Returns:
            A scalar loss value.
        """
        device = features.device  # Detect if using CUDA or CPU

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...].')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`.')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features.')
            mask = torch.eq(labels, labels.T).float().to(device)

            # Apply hierarchy weights dynamically
            if self.hierarchy:
                for parent, related in self.hierarchy.items():
                    for child, weight in related.items():
                        child_indices = (labels == child).float().to(device)
                        parent_indices = (labels == parent).float().to(device)
                        mask += torch.mm(parent_indices, child_indices.T) * weight

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Compute final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Custom dataset to load images from a given folder.
        Args:
            folder_path: Path to the folder containing images.
            transform: Transformations to apply to the images.
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [folder_path + "/1/"+ file for file in os.listdir(folder_path + "/1/") if ".ipynb" not in file] + [folder_path + "/0/" + file for file in os.listdir(folder_path + "/0/") if ".ipynb" not in file]
        self.labels = [1 for file in os.listdir(folder_path + "/1/") if ".ipynb" not in file] + [0 for file in os.listdir(folder_path + "/0/") if ".ipynb" not in file]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path)#.convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def generate_random_views(image):
    """
    Generate two random views of an image using horizontal, vertical, and rotational shifts.
    Args:
        image: A single image tensor.
    Returns:
        A tuple of two randomly transformed images.
    """
    random_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])
    view1 = random_transforms(image)
    view2 = random_transforms(image)
    return view1, view2

class RandomViewDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        view1, view2 = generate_random_views(image)
        return (view1, view2), label

def get_dataloader(train_folder_path, batch_size=32):
    """
    Create a DataLoader for the training dataset with random views.
    Args:
        train_folder_path: Path to the training dataset folder.
        batch_size: Batch size for the DataLoader.
    Returns:
        DataLoader for the training dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    base_dataset = CustomDataset(folder_path=train_folder_path, transform=transform)
    random_view_dataset = RandomViewDataset(base_dataset)
    data_loader = DataLoader(random_view_dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class Encoder(nn.Module):
    def __init__(self, input_channels=1):
        """
        Builds a simple encoder network.
        Args:
            input_channels: Number of channels in the input image (default is 1 for grayscale).
            embedding_dim: Dimension of the output embedding.
        """
        super(Encoder, self).__init__()
        
        # Convolutional layers with pooling
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.5)

        # AdaptiveAvgPool2d to output (1, 1) spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.tanh(self.conv2(x)))
        x = F.relu(self.conv2(x))

        # Average pooling for embeddings
        x = self.avg_pool(x)

        # Flatten to create the final embeddings
        x = x.view(x.size(0), -1)
        return x
    
    
    
class MLPProjection(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64):
        """
        Builds an MLP projection model with two linear layers.
        Args:
            input_dim: Dimension of the input vector (default is 128).
            hidden_dim: Number of nodes in the hidden layer (default is 128).
            output_dim: Dimension of the output vector (default is 128).
        """
        super(MLPProjection, self).__init__()
        
        # First linear layer with hidden_dim nodes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Second linear layer with output_dim nodes
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply first linear layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply second linear layer
        x = self.fc2(x)
        return x
    
    
    
train_folder_path = "/home/yaganapu/senescence/data/updated_model_data/whole_image/contrastive_dataset_overlapping_patches/train"  # Replace with your train folder path
batch_size = 64

train_loader = get_dataloader(train_folder_path, batch_size)
print(f"Train DataLoader ready with {len(train_loader)} batches.")

val_folder_path = "/home/yaganapu/senescence/data/updated_model_data/whole_image/contrastive_dataset_overlapping_patches/val"  # Replace with your train folder path
batch_size = 64

val_loader = get_dataloader(val_folder_path, batch_size)
print(f"Val DataLoader ready with {len(val_loader)} batches.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(input_channels=1).to(device)

mlp = MLPProjection(input_dim=64, hidden_dim=64, output_dim=64).to(device)

criterion = EmpiricalSupConLoss(hierarchy=hierarchy)

# for views, labels in train_loader:
#     view1, view2 = views
#     print("View 1 shape:", view1.shape)
#     print("View 2 shape:", view2.shape)
    
#     # Forward pass
#     embeddings1, embeddings2 = encoder(view1), encoder(view2)
    
#     # Forward pass through MLP projection
#     projected_embeddings1, projected_embeddings2 = mlp(embeddings1), mlp(embeddings2)
    
#     print("MLP Projection Output shape:", projected_embeddings1.shape, projected_embeddings2.shape)
    
#     combined_tensor = torch.stack((projected_embeddings1, projected_embeddings2), dim=1)
    
#     print(loss(combined_tensor, labels))
#     break

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mlp.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

train_loss_values, val_loss_values = [], []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    encoder.train()
    mlp.train()
    running_loss = 0.0
    num_batches = 0
    
    for views, labels in train_loader:
        num_batches += 1
        view1, view2 = views
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        # Forward pass
        embeddings1, embeddings2 = encoder(view1), encoder(view2)
        projected_embeddings1, projected_embeddings2 = mlp(embeddings1), mlp(embeddings2)

        combined_tensor = torch.stack((projected_embeddings1, projected_embeddings2), dim=1)

        # Calculate loss
        loss = criterion(combined_tensor, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss_values.append(running_loss / num_batches)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

    encoder.eval()
    mlp.eval()
    running_loss = 0.0
    num_batches = 0
    for views, labels in val_loader:
        num_batches += 1
        view1, view2 = views
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

        # Forward pass
        embeddings1, embeddings2 = encoder(view1), encoder(view2)
        projected_embeddings1, projected_embeddings2 = mlp(embeddings1), mlp(embeddings2)

        combined_tensor = torch.stack((projected_embeddings1, projected_embeddings2), dim=1)

        # Calculate loss
        loss = criterion(combined_tensor, labels)

        running_loss += loss.item()
    
    val_loss_values.append(running_loss / num_batches)
    scheduler.step(val_loss_values[-1])

    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {running_loss:.4f}")
    
# Save the trained encoder
encoder_save_path = "/home/yaganapu/senescence/contrastive_learning_overlapping_patches/embeddings_overlapping_patches_64dim_t1/encoder_overlapping_patches_64dim.pth"  # Define the path to save the encoder
torch.save(encoder.state_dict(), encoder_save_path)
print(f"Encoder saved to {encoder_save_path}")

def plot_tsne(embeddings, all_labels, save_path, num_classes=2):
    """
    Apply t-SNE to embeddings, plot, and save the scatter plot.
    Args:
        embeddings: 2D or higher-dimensional data (numpy array).
        all_labels: List or array of labels corresponding to embeddings.
        save_path: Path to save the plot.
        num_classes: Number of unique classes for coloring.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Generate scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=all_labels,
        cmap=plt.get_cmap("tab10", num_classes),
        s=10,
        alpha=0.8,
    )
    plt.colorbar(scatter, ticks=range(num_classes), label="Class Labels")
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)

    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")
    
    
def save_embeddings_and_labels(encoder, data_loader, device, save_path):
    """
    Save embeddings and labels for a dataset using the encoder and generate t-SNE plots.
    Args:
        encoder: The encoder model.
        data_loader: DataLoader for the dataset.
        device: Device to run the computation (CPU or GPU).
        save_path: Path to save the embeddings, labels, and t-SNE plot.
    """
    os.makedirs(save_path, exist_ok = True)
    encoder.eval()  # Set encoder to evaluation mode
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for views, labels in data_loader:
            views = views.to(device)
            embeddings = encoder(views)
            
            # Collect embeddings and labels
            all_embeddings.extend(embeddings.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save embeddings and labels
    embeddings_path = os.path.join(save_path, "embeddings.npy")
    labels_path = os.path.join(save_path, "labels.npy")
    tsne_plot_path = os.path.join(save_path, "tsne_plot.png")

    np.save(embeddings_path, np.array(all_embeddings))
    np.save(labels_path, np.array(all_labels))
    print(f"Saved embeddings and labels to {save_path}")

    # Generate and save t-SNE plot
    plot_tsne(np.array(all_embeddings), np.array(all_labels), save_path=tsne_plot_path, num_classes=2)
    
    
    
train_folder_path = "/home/yaganapu/senescence/data/updated_model_data/whole_image/contrastive_dataset_overlapping_patches/train"
val_folder_path = "/home/yaganapu/senescence/data/updated_model_data/whole_image/contrastive_dataset_overlapping_patches/val"
test_folder_path = "/home/yaganapu/senescence/data/updated_model_data/whole_image/contrastive_dataset_overlapping_patches/test"

save_dir = "/home/yaganapu/senescence/contrastive_learning_overlapping_patches/embeddings_overlapping_patches_64dim_t1/embeddings"  # Directory to save embeddings and labels
os.makedirs(save_dir, exist_ok=True)

batch_size = 64
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])


    
# Prepare datasets and loaders
train_dataset = CustomDataset(folder_path=train_folder_path, transform=transform)
val_dataset = CustomDataset(folder_path=val_folder_path, transform=transform)
test_dataset = CustomDataset(folder_path=test_folder_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
# Save embeddings and labels for each dataset
save_embeddings_and_labels(encoder, train_loader, device, os.path.join(save_dir, "train"))
save_embeddings_and_labels(encoder, val_loader, device, os.path.join(save_dir, "val"))
save_embeddings_and_labels(encoder, test_loader, device, os.path.join(save_dir, "test"))



import matplotlib.pyplot as plt

def plot_loss_curve(train_loss_values, val_loss_values, save_path):
    """
    Plot and save the training and validation loss curves.
    Args:
        train_loss_values: List of training loss values over epochs.
        val_loss_values: List of validation loss values over epochs.
        save_path: Path to save the loss curve plot.
    """
    epochs = range(1, len(train_loss_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_values, label="Training Loss", marker='o')
    plt.plot(epochs, val_loss_values, label="Validation Loss", marker='o')

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

# Save the loss plot
loss_plot_path = "/home/yaganapu/senescence/contrastive_learning_overlapping_patches/embeddings_overlapping_patches_64dim_t1/train_val_loss_curve.png"
plot_loss_curve(train_loss_values, val_loss_values, save_path=loss_plot_path)





    
    
    
   