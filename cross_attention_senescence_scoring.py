import argparse
import os


from lion_pytorch import Lion

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Add argument parsing
parser = argparse.ArgumentParser(description="Run contrastive learning with configurable settings.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation.")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to be made visible.")
parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss.")
parser.add_argument("--embedding_dim", type=int, default=256, help="embedding dim for contrastive loss.")
parser.add_argument("--iteration", type=int, required=True, help="Iteration value for different dataset splits.")
parser.add_argument("--hierarchy_weight", type=float, default=0.1, help="Weight for hierarchical similarity in contrastive loss.")
args = parser.parse_args()

# Set GPU ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

print(f"Using GPU ID: {args.gpu_id}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Use the parsed batch size and temperature
batch_size = args.batch_size
temperature = args.temperature
embedding_dim = args.embedding_dim
iteration = args.iteration

hierarchy_weight = args.hierarchy_weight

print(f"Batch size: {batch_size}")
print(f"Temperature: {temperature}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Iteration: {iteration}")
print(f"Hierarchy Weight: {hierarchy_weight}")


# Define hierarchy dynamically using the hierarchy weight
hierarchy = {
    0: {1: hierarchy_weight},  # Control nucleus ↔ Control non-nucleus closer, minimal relationship with Stress
    1: {0: hierarchy_weight},
    2: {3: hierarchy_weight},  # Stress nucleus ↔ Stress non-nucleus closer, minimal relationship with Control
    3: {2: hierarchy_weight}
}

print(hierarchy)




import torch.optim.swa_utils as swa_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
import pandas as pd

import os
import cv2
import numpy as np
from skimage.morphology import disk, binary_dilation, binary_erosion, binary_opening, remove_small_objects
from skimage.segmentation import clear_border
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.measure import label as importlabel
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from PIL import Image
from matplotlib.patches import Rectangle


import pandas as pd


print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")


# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# # Function to plot training and validation loss
# def plot_loss_curve(train_loss_values, val_loss_values, num_epochs=None):
#     if num_epochs is None:
#         num_epochs = len(train_loss_values)  # Default to the length of train_loss_values
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, num_epochs + 1), train_loss_values, label='Training Loss')
#     plt.plot(range(1, num_epochs + 1), val_loss_values, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/swa_{batch_size}_{embedding_dim}_{temperature}.jpg")

# Function to find optimal threshold
def find_optimal_threshold(labels, predictions):
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_f1, optimal_threshold = 0, 0
    threshold_metrics = []

    for t in thresholds:
        predicted_labels = (predictions >= t).astype(int)
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)
        threshold_metrics.append((t, precision, recall, f1))

        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = t

    return optimal_threshold, best_f1, threshold_metrics

# Function to calculate evaluation metrics
def calculate_metrics(labels, predictions, threshold):
    preds = (predictions >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc_score = roc_auc_score(labels, predictions)
    precisions, recalls, _ = precision_recall_curve(labels, predictions)
    auc_pr = auc(recalls, precisions)
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': auc_score,
        'AUC-PR': auc_pr,
        'Optimal Threshold': threshold
    }


    
    
class PatchExtractor:
    def __init__(self, patch_size=50, overlap_step=25, min_coverage=0.3, non_nucleus_overlap_step=25):
        self.patch_size = patch_size
        self.overlap_step = overlap_step
        self.min_coverage = min_coverage
        self.non_nucleus_overlap_step = non_nucleus_overlap_step
        self.overlap_step_nucleus = overlap_step
        self.overlap_step_non_nucleus = non_nucleus_overlap_step

    def preprocess_image(self, image):
        """
        Preprocess the image to extract the binary mask of the nucleus.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_adjusted = clahe.apply(image)
        filtered = cv2.bilateralFilter(contrast_adjusted, d=9, sigmaColor=75, sigmaSpace=75)
        _, binary_threshold = cv2.threshold(filtered, 128, 255, cv2.THRESH_BINARY)
        filled_image = binary_fill_holes(binary_threshold > 0)
        dilated = binary_dilation(filled_image, disk(15))
        eroded = binary_erosion(dilated, disk(3))
        opened = binary_opening(eroded, disk(5))
        cleaned_mask = remove_small_objects(opened, min_size=2000).astype(np.uint8)
        return cleaned_mask

    def extract_patches(self, image_path, all_visualize=False):
        """
        Extract nucleus and non-nucleus patches from a single image and visualize/save them.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2:
            print(f"Skipping {image_path}, not a valid grayscale image.")
            return

        # Step 1: Preprocess the image
        binary_mask = self.preprocess_image(image)

        # Step 2: Identify the largest component (nucleus)
        labeled_image = label(binary_mask)
        regions = regionprops(labeled_image)
        if not regions:
            print(f"No nucleus found in {image_path}")
            return

        # Get the largest region (nucleus)
        nucleus_region = max(regions, key=lambda r: r.area)
        y_centroid, x_centroid = map(int, nucleus_region.centroid)

        # Step 3: Extract nucleus patches (3x3 grid around centroid) with coverage check
        h, w = image.shape
        half_patch = self.patch_size // 2
        
        # Nucleus patches (9-grid)
        nucleus_patches = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                y_start = max(0, y_centroid + dy * self.patch_size - half_patch)
                x_start = max(0, x_centroid + dx * self.patch_size - half_patch)
                y_end = min(y_start + self.patch_size, h)
                x_end = min(x_start + self.patch_size, w)
                patch = image[y_start:y_end, x_start:x_end]
                patch_mask = binary_mask[y_start:y_end, x_start:x_end]
                coverage = np.sum(patch_mask) / (self.patch_size * self.patch_size)
                nucleus_patches.append(((x_start, y_start, x_end, y_end), patch, coverage))

        # Overlapping nucleus patches
        overlapping_patches = []
        for (x_start, y_start, x_end, y_end), _, _ in nucleus_patches:
            for oy in range(y_start, y_end - self.overlap_step_nucleus + 1, self.overlap_step_nucleus):
                for ox in range(x_start, x_end - self.overlap_step_nucleus + 1, self.overlap_step_nucleus):
                    sub_x_end = min(ox + self.patch_size, x_end)
                    sub_y_end = min(oy + self.patch_size, y_end)
                    patch = image[oy:sub_y_end, ox:sub_x_end]
                    patch_mask = binary_mask[oy:sub_y_end, ox:sub_x_end]
                    coverage = np.sum(patch_mask) / (self.patch_size * self.patch_size)
                    if coverage >= self.min_coverage:
                        overlapping_patches.append(((ox, oy, sub_x_end, sub_y_end), patch, coverage))

        # Non-nucleus patches
        non_nucleus_patches = []
        for y in range(0, h - self.patch_size + 1, self.patch_size):
            for x in range(0, w - self.patch_size + 1, self.patch_size):
                if any(
                    x_start < x + self.patch_size and x < x_end and y_start < y + self.patch_size and y < y_end
                    for (x_start, y_start, x_end, y_end), _, _ in nucleus_patches
                ):
                    continue
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                patch_mask = binary_mask[y:y + self.patch_size, x:x + self.patch_size]
                coverage = np.sum(patch_mask) / (self.patch_size * self.patch_size)
                if coverage >= self.min_coverage:
                    non_nucleus_patches.append(((x, y), patch, coverage))

        # Overlapping non-nucleus patches
        overlapping_non_nucleus_patches = []
        for y in range(0, h - self.patch_size + 1, self.overlap_step_non_nucleus):
            for x in range(0, w - self.patch_size + 1, self.overlap_step_non_nucleus):
                if any(
                    x_start < x + self.patch_size and x < x_end and y_start < y + self.patch_size and y < y_end
                    for (x_start, y_start, x_end, y_end), _, _ in nucleus_patches
                ):
                    continue
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                patch_mask = binary_mask[y:y + self.patch_size, x:x + self.patch_size]
                coverage = np.sum(patch_mask) / (self.patch_size * self.patch_size)
                if coverage >= self.min_coverage:
                    overlapping_non_nucleus_patches.append(((x, y), patch, coverage))



        

        # Step 7: Visualize or Save Results
        if all_visualize:
            self.visualize_patches(image, binary_mask, nucleus_patches, overlapping_patches, non_nucleus_patches, overlapping_non_nucleus_patches)
            
        return overlapping_patches, overlapping_non_nucleus_patches

    def visualize_patches(self, image, binary_mask, nucleus_patches, overlapping_patches, non_nucleus_patches, overlapping_non_nucleus_patches):
        """
        Visualize nucleus and non-nucleus patches and overlay them on the original image.
        """
        # Display binary mask
        plt.figure(figsize=(8, 8))
        plt.title("Binary Mask")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        plt.show()

        # Nucleus 9-grid patches visualization with coverage annotation
        overlay_image_9grid = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x_start, y_start, x_end, y_end), _, coverage in nucleus_patches:
            color = (255, 0, 0)  # Blue for nucleus grid
            cv2.rectangle(overlay_image_9grid, (x_start, y_start), (x_end, y_end), color, 2)
            if coverage < self.min_coverage:
                cv2.putText(overlay_image_9grid, "Excluded", (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        plt.figure(figsize=(8, 8))
        plt.title("Nucleus 9-Grid Patches (Blue)")
        plt.imshow(cv2.cvtColor(overlay_image_9grid, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Overlapping nucleus patches visualization
        overlay_image_overlapping = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x_start, y_start, x_end, y_end), _, _ in overlapping_patches:
            cv2.rectangle(overlay_image_overlapping, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        plt.figure(figsize=(8, 8))
        plt.title("Overlapping Nucleus Patches (Green)")
        plt.imshow(cv2.cvtColor(overlay_image_overlapping, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Non-nucleus patches visualization
        overlay_image_non_nucleus = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y), _, _ in non_nucleus_patches:
            cv2.rectangle(overlay_image_non_nucleus, (x, y), (x + self.patch_size, y + self.patch_size), (0, 0, 255), 2)
        plt.figure(figsize=(8, 8))
        plt.title("Non-Nucleus Patches (Red)")
        plt.imshow(cv2.cvtColor(overlay_image_non_nucleus, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Overlapping non-nucleus patches visualization
        overlay_image_overlapping_non_nucleus = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x_start, y_start, x_end, y_end), _, _ in overlapping_non_nucleus_patches:
            cv2.rectangle(overlay_image_overlapping_non_nucleus, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
        plt.figure(figsize=(8, 8))
        plt.title("Overlapping Non-Nucleus Patches (Yellow)")
        plt.imshow(cv2.cvtColor(overlay_image_overlapping_non_nucleus, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        
class ImageDataset(Dataset):
    def __init__(self, folder_path, patch_extractor, transform=None, model=None, device=None, visualize=False):
        """
        Args:
            folder_path: Path to the root directory containing 'control' and 'stress' subdirectories.
            patch_extractor: Instance of the PatchExtractor class.
            transform: PyTorch transform for preprocessing patches.
            model: Trained model to predict patch scores.
            device: Device for model inference.
            visualize: Whether to visualize top 10 patches on the original image.
        """
        self.folder_path = folder_path
        self.patch_extractor = patch_extractor
        self.transform = transform
        self.model = model
        self.device = device
        self.visualize = visualize

        self.image_files = []
        self.labels = []

        # Process images in the `control` and `stress` subdirectories
        for sub_dir in ["control", "stress"]:
            full_path = os.path.join(folder_path, sub_dir)
            if not os.path.exists(full_path):
                print(f"Warning: {full_path} does not exist. Skipping.")
                continue
            for file in os.listdir(full_path):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(full_path, file)
                    #print(f"Processing image: {img_path}")
                    try:
                        nucleus_patches, non_nucleus_patches = self.patch_extractor.extract_patches(img_path)
                        if len(nucleus_patches) < 10 or len(non_nucleus_patches) < 10:
                            print(f"Skipping {img_path} - Insufficient patches: "
                                  f"Nucleus = {len(nucleus_patches)}, Non-Nucleus = {len(non_nucleus_patches)}")
                            continue
                        self.image_files.append(img_path)
                        real_label = 0 if "control" in sub_dir.lower() else 1
                        self.labels.append(real_label)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

        # Process images directly in the root directory (unknown labels)
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')) and \
               not any(folder in full_path for folder in ["control", "stress"]):
                #print(f"Processing unknown-label image: {full_path}")
                try:
                    nucleus_patches, non_nucleus_patches = self.patch_extractor.extract_patches(full_path)
                    if len(nucleus_patches) < 10 or len(non_nucleus_patches) < 10:
                        print(f"Skipping {full_path} - Insufficient patches: "
                              f"Nucleus = {len(nucleus_patches)}, Non-Nucleus = {len(non_nucleus_patches)}")
                        continue
                    self.image_files.append(full_path)
                    self.labels.append(2)  # Unknown label
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

        #print(f"Total images loaded: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            true_label = self.labels[idx]

            #print(f"Loading image at index {idx}: {img_path}")
            # Extract patches using the PatchExtractor
            nucleus_patches, non_nucleus_patches = self.patch_extractor.extract_patches(img_path)

            #print(f"Nucleus patches: {len(nucleus_patches)}, Non-Nucleus patches: {len(non_nucleus_patches)}")
            if len(nucleus_patches) < 10 or len(non_nucleus_patches) < 10:
                print(f"Skipping image at index {idx} due to insufficient patches.")
                return None

            # Predict scores for nucleus patches
            nucleus_predictions = []
            for (x_start, y_start, x_end, y_end), patch, _ in nucleus_patches:
                patch_tensor = self.transform(Image.fromarray(patch)).unsqueeze(0).to(self.device)
                prediction = self.model(patch_tensor).item()
                nucleus_predictions.append(((x_start, y_start, x_end, y_end), patch_tensor, prediction))

            # Predict scores for non-nucleus patches
            non_nucleus_predictions = []
            for (x, y), patch, _ in non_nucleus_patches:
                patch_tensor = self.transform(Image.fromarray(patch)).unsqueeze(0).to(self.device)
                prediction = self.model(patch_tensor).item()
                non_nucleus_predictions.append(((x, y), patch_tensor, prediction))

            # Sort and select top 10 nucleus and non-nucleus patches
            top_nucleus = sorted(nucleus_predictions, key=lambda x: x[2], reverse=True)[:10]
            top_non_nucleus = sorted(non_nucleus_predictions, key=lambda x: x[2], reverse=True)[:10]

            #print(f"Top 10 Nucleus predictions: {[x[2] for x in top_nucleus]}")
            #print(f"Top 10 Non-Nucleus predictions: {[x[2] for x in top_non_nucleus]}")

            # Combine tensors for top patches
            top_nucleus_tensors = torch.cat([x[1] for x in top_nucleus], dim=0)
            top_non_nucleus_tensors = torch.cat([x[1] for x in top_non_nucleus], dim=0)

            return {
                "image_idx": idx,
                "nucleus_patches": top_nucleus_tensors,
                "nucleus_probs": [x[2] for x in top_nucleus],
                "non_nucleus_patches": top_non_nucleus_tensors,
                "non_nucleus_probs": [x[2] for x in top_non_nucleus],
                "true_label": true_label
            }
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            return None



    @staticmethod
    def visualize_top_patches(image, patches, title, color="blue"):
        """
        Visualize the top patches by overlaying their bounding boxes on the image.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='gray')
        for (x_start, y_start, x_end, y_end), _, _ in patches:
            rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                             linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        ax.set_title(title)
        plt.axis('off')
        plt.show()
        
        
def create_dataloader(folder_path, patch_extractor, transform, model, device, batch_size=1, visualize=False):
    """
    Create a DataLoader for a given dataset.

    Args:
        folder_path: Path to the dataset directory.
        patch_extractor: Instance of the PatchExtractor class.
        transform: PyTorch transform for preprocessing patches.
        model: Trained model to predict patch scores.
        device: Device for model inference.
        batch_size: Batch size for the DataLoader.
        visualize: Whether to visualize patches.

    Returns:
        DataLoader object.
    """
    dataset = ImageDataset(
        folder_path=folder_path,
        patch_extractor=patch_extractor,
        transform=transform,
        model=model,
        device=device,
        visualize=visualize
    )
    return DataLoader(dataset, batch_size=1, shuffle=True)



class Encoder(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=embedding_dim):
        super(Encoder, self).__init__()
        layers = []
        channels = [input_channels, 32, 64, 128, 256][:int(np.log2(embedding_dim / 32)) + 2]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if i != len(channels) - 2:  # No pooling in the last layer
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.network = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.network(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)
    

    
    
    
    
    
class MLPProjection(nn.Module):
    def __init__(self, input_dim=embedding_dim, hidden_dim=embedding_dim, output_dim=embedding_dim):
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
class BinaryClassifier(nn.Module):
    def __init__(self, encoder, mlp_head, embedding_dim=embedding_dim):
        """
        A binary classifier model that uses a frozen encoder and MLP head for embeddings.
        Args:
            encoder: The frozen encoder model.
            mlp_head: The frozen MLP head model.
            embedding_dim: The dimension of the MLP head's output embeddings.
        """
        super(BinaryClassifier, self).__init__()

        # Freeze the encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Freeze the MLP head parameters
        for param in mlp_head.parameters():
            param.requires_grad = False

        self.encoder = encoder
        self.mlp_head = mlp_head
        self.fc = nn.Linear(embedding_dim, 1)  # Fully connected layer for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output

    def forward(self, x):
        with torch.no_grad():  # Ensure encoder and MLP head are not updated
            features = self.encoder(x)  # Extract features using the frozen encoder
            embeddings = self.mlp_head(features)  # Get embeddings from the MLP head
            embeddings = F.normalize(embeddings)

        logits = self.fc(embeddings)  # Pass embeddings through the fully connected layer
        output = self.sigmoid(logits)  # Apply sigmoid to get probabilities
        return output

# Initialize PatchExtractor and transformations
patch_extractor = PatchExtractor(patch_size=50)
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_channels=1).to(device)
mlp_head = MLPProjection(input_dim=embedding_dim, hidden_dim=embedding_dim, output_dim=embedding_dim).to(device)

binary_model = BinaryClassifier(encoder, mlp_head, embedding_dim=embedding_dim).to(device)

encoder_save_path = f"/home/yaganapu/senescence/new_generated_data/pervious_model_with_dfloss/contrastive_with_hs08/contrastive_learning_another_try_500_with_adamw/on_total_data/1000epoch_decay_embeddings_overlapping_patches_64dim_{batch_size}_{temperature}_{embedding_dim}_h{hierarchy_weight}/iteration_{iteration}/encoder_64_{batch_size}_{temperature}_{embedding_dim}_{iteration}.pth"
print(encoder_save_path)
# Load the encoder with mapping to the appropriate device
encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
encoder.eval()  # Ensure the encoder is in evaluation mode

print(f"Encoder loaded successfully on {device}.")






mlp_save_path = f"/home/yaganapu/senescence/new_generated_data/pervious_model_with_dfloss/contrastive_with_hs08/contrastive_learning_another_try_500_with_adamw/on_total_data/1000epoch_decay_embeddings_overlapping_patches_64dim_{batch_size}_{temperature}_{embedding_dim}_h{hierarchy_weight}/iteration_{iteration}/head_64_{batch_size}_{temperature}_{embedding_dim}_{iteration}.pth"
print(mlp_save_path)
# Load the encoder with mapping to the appropriate device
mlp_head.load_state_dict(torch.load(mlp_save_path, map_location=device))
mlp_head.eval()  # Ensure the encoder is in evaluation mode

print(f"mlp loaded successfully on {device}.")



# encoder_path = "/home/yaganapu/senescence/contrastive_learning_overlapping_patches/embeddings_overlapping_patches_64dim_t07/encoder_overlapping_patches_64dim.pth"
model_path = f"/home/yaganapu/senescence/new_generated_data/pervious_model_with_dfloss/contrastive_with_hs08/contrastive_learning_another_try_500_with_adamw/on_total_data/binary_classifier/1000epoch_swa_binary_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/binary_model_{batch_size}_{embedding_dim}_{temperature}_{iteration}.pth"
print(model_path)


saved_state_dict = torch.load(model_path, map_location=device)
# Remove the "n_averaged" key and strip "module." prefix
new_state_dict = {k.replace("module.", ""): v for k, v in saved_state_dict.items() if k != "n_averaged"}

#encoder.load_state_dict(torch.load(encoder_path, map_location=device))
binary_model.load_state_dict(new_state_dict)

print(f"binary model loaded successfully on {device}.")

encoder.eval()
mlp_head.eval()
binary_model.eval()


print(encoder.eval())
print(mlp_head.eval())
#print(binary_model.eval())

# Create DataLoader for train/val/test
train_loader = create_dataloader(
    folder_path=f"/home/yaganapu/senescence/data/updated_model_data/whole_image/overlapping_contrastive_dataset_patches",
    patch_extractor=patch_extractor,
    transform=transform,
    model=binary_model,
    device=device,
    batch_size=1,
    visualize=False  # Set True to visualize top 10 patches
)



# # Validation DataLoader
# val_loader = create_dataloader(
#     folder_path=f"/home/yaganapu/senescence/data/updated_model_data/whole_image/iterations_for_proposed_model/iteration_{iteration}/val",
#     patch_extractor=patch_extractor,
#     transform=transform,
#     model=binary_model,
#     device=device,
#     batch_size=1,
#     visualize=False  # Set to True for visualization during validation
# )

# # Test DataLoader
# test_loader = create_dataloader(
#     folder_path=f"/home/yaganapu/senescence/data/updated_model_data/whole_image/iterations_for_proposed_model/iteration_{iteration}/test",
#     patch_extractor=patch_extractor,
#     transform=transform,
#     model=binary_model,
#     device=device,
#     batch_size=1,
#     visualize=False  # Set True for test data to visualize top patches
# )

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
#         super(ProjectionHead, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.scaling_factor = embed_dim ** 0.5
        # Learnable parameter, initially set to 0.0 (sigmoid(0) = 0.5)
        #self.raw_lambda_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, queries, keys, values):
        Q = self.Wq(queries)  # (B, N, D)
        K = self.Wk(keys)    # (B, M, D)
        V = self.Wv(values)  # (B, M, D)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scaling_factor   # (B, N, M)
        weights = F.softmax(scores, dim=-1)  # (B, N, M)

        # Attention output
        attention_output = torch.matmul(weights, V)  # (B, N, D)
        
        
        # Apply sigmoid to map raw_lambda_param to [0, 1]
        #lambda_param = torch.sigmoid(self.raw_lambda_param)

        # Refined features: X' = X + \lambda * Attention(X, Y)
        #refined_queries = queries + lambda_param * attention_output
        return attention_output
    
    
    
    
class BinaryPredictionHead(nn.Module):
    def __init__(self, input_dim):
        super(BinaryPredictionHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Input `x` is the concatenated scores of nucleus and non-nucleus patches
        return torch.sigmoid(self.fc(x))
    
    
class CrossAttentionModel(nn.Module):
    def __init__(self, encoder, mlp_head, embed_dim=embedding_dim):
        super(CrossAttentionModel, self).__init__()

        # Frozen encoder
        self.encoder = encoder
        self.mlp_head = mlp_head
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Freeze the MLP head parameters
        for param in mlp_head.parameters():
            param.requires_grad = False

        self.cross_attention_nucleus = CrossAttention(embed_dim)  # Nucleus -> Non-Nucleus
        self.cross_attention_non_nucleus = CrossAttention(embed_dim)  # Non-Nucleus -> Nucleus

        # Raw learnable parameter for alpha
        # self.raw_alpha1 = nn.Parameter(torch.tensor(0.0))
        # self.raw_alpha2 = nn.Parameter(torch.tensor(0.0))
        # self.raw_alpha3 = nn.Parameter(torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))# Initial value corresponds to sigmoid(0) = 0.5

        # Linear layer for patch-wise projection
        self.patch_linear = nn.Linear(embed_dim, 1)

        # Binary prediction head
        self.binary_head = BinaryPredictionHead(20)

    def forward(self, nucleus_patches, non_nucleus_patches):
        # Encode patches
        B, N, C, H, W = nucleus_patches.shape  # Batch size, number of nucleus patches
        _, M, _, _, _ = non_nucleus_patches.shape  # Number of non-nucleus patches

        # Reshape patches into batches
        nucleus_patches = nucleus_patches.view(B * N, C, H, W)
        non_nucleus_patches = non_nucleus_patches.view(B * M, C, H, W)

        # Encode patches
        nucleus_embeddings_features = self.encoder(nucleus_patches)  # (B * N, D)
        non_nucleus_embeddings_features = self.encoder(non_nucleus_patches)  # (B * M, D)
        nucleus_embeddings = self.mlp_head(nucleus_embeddings_features)  # (B * N, D)
        non_nucleus_embeddings = self.mlp_head(non_nucleus_embeddings_features)  # (B * M, D)

        # Reshape back to batch with patches
        nucleus_proj = nucleus_embeddings.view(B, N, -1)  # (B, N, D)
        non_nucleus_proj = non_nucleus_embeddings.view(B, M, -1)  # (B, M, D)

        # Cross-attention: Nucleus -> Non-Nucleus
        attention_nucleus = self.cross_attention_nucleus(
            nucleus_proj, non_nucleus_proj, nucleus_proj
        )  # (B, N, D)

        # Cross-attention: Non-Nucleus -> Nucleus
        attention_non_nucleus = self.cross_attention_non_nucleus(
            non_nucleus_proj, nucleus_proj, non_nucleus_proj
        )  # (B, M, D)

        # Process raw_alpha through sigmoid to ensure it is in [0, 1]
        alpha = torch.sigmoid(self.raw_alpha)

        refined_nucleus = nucleus_proj + alpha * attention_nucleus
        refined_non_nucleus = non_nucleus_proj + (1 - alpha) * attention_non_nucleus

        # Stack refined nucleus and non-nucleus features along the patch dimension
        combined_features = torch.cat([refined_nucleus, refined_non_nucleus], dim=1)  # (B, 20, D)

        # Pass through linear layer for each patch
        patch_scores = self.patch_linear(combined_features)  # (B, 20, 1)

        # Squeeze to shape (B, 20)
        patch_scores = patch_scores.squeeze(-1)  # (B, 20)

        # Binary classification
        output = self.binary_head(patch_scores)

        return output, attention_nucleus, attention_non_nucleus, alpha
    

    

cross_attention_model = CrossAttentionModel(encoder, mlp_head,  embed_dim=embedding_dim).to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

optimizer = optim.Adam(cross_attention_model.parameters(), lr=0.001,)
#optimizer = Lion(cross_attention_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True) 


num_epochs = 100
train_loss_values, val_loss_values = [], []



# # Initialize SWA components
# swa_model = swa_utils.AveragedModel(cross_attention_model).to(device)
# swa_start_epoch = 80  # Epoch to start SWA
# #swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
# swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - swa_start_epoch, eta_min=1e-5)





for epoch in range(num_epochs):
    # Training phase
    cross_attention_model.train()
    train_running_loss = 0.0
    train_batches = 0
    all_train_labels, all_train_predictions = [], []

    for batch in train_loader:
        nucleus_patches = batch['nucleus_patches'].to(device)
        non_nucleus_patches = batch['non_nucleus_patches'].to(device)
        labels = torch.tensor([patch.squeeze(0) for patch in batch['true_label']], dtype=torch.float).to(device)

        # # Add batch dimension
        # nucleus_patches = nucleus_patches.unsqueeze(0)  # (1, 10, 1, 50, 50)
        # non_nucleus_patches = non_nucleus_patches.unsqueeze(0)  # (1, 10, 1, 50, 50)

        train_batches += 1

        # Forward pass
        outputs, _, _,_  = cross_attention_model(nucleus_patches, non_nucleus_patches)
        outputs = outputs.squeeze(dim=-1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        all_train_labels.extend(labels.cpu().numpy())
        all_train_predictions.extend(outputs.detach().cpu().numpy())

    train_loss = train_running_loss / train_batches
    train_loss_values.append(train_loss)

    # Calculate training AUC and AUC-PR
    train_auc = roc_auc_score(all_train_labels, all_train_predictions)
    precisions, recalls, _ = precision_recall_curve(all_train_labels, all_train_predictions)
    train_auc_pr = auc(recalls, precisions)

#     # Validation phase
#     cross_attention_model.eval()
#     val_running_loss = 0.0
#     val_batches = 0
#     all_val_labels, all_val_predictions = [], []

#     with torch.no_grad():
#         for batch in val_loader:
#             nucleus_patches = torch.stack([patch.squeeze(0) for patch in batch['nucleus_patches']]).to(device)
#             non_nucleus_patches = torch.stack([patch.squeeze(0) for patch in batch['non_nucleus_patches']]).to(device)
#             labels = torch.tensor([patch.squeeze(0) for patch in batch['true_label']], dtype=torch.float).to(device)

#             # # Add batch dimension
#             # nucleus_patches = nucleus_patches.unsqueeze(0)
#             # non_nucleus_patches = non_nucleus_patches.unsqueeze(0)

#             val_batches += 1

#             # Forward pass
#             outputs, _, _,_  = cross_attention_model(nucleus_patches, non_nucleus_patches)
#             outputs = outputs.squeeze(dim=-1)
#             loss = criterion(outputs, labels)

#             val_running_loss += loss.item()
#             all_val_labels.extend(labels.cpu().numpy())
#             all_val_predictions.extend(outputs.cpu().numpy())

#     val_loss = val_running_loss / val_batches
#     val_loss_values.append(val_loss)

#     # Calculate validation AUC and AUC-PR
#     val_auc = roc_auc_score(all_val_labels, all_val_predictions)
#     precisions, recalls, _ = precision_recall_curve(all_val_labels, all_val_predictions)
#     val_auc_pr = auc(recalls, precisions)
    
    
#     # if epoch >= swa_start_epoch:
#     #     swa_model.update_parameters(cross_attention_model)
#     #     swa_scheduler.step()  # Step cosine scheduler
#     # else:
#     #       scheduler.step(val_loss_values[-1])

    scheduler.step(train_loss)

    # Print metrics for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Train AUC: {train_auc:.4f} | Train AUC-PR: {train_auc_pr:.4f}")
    #print(f"Val AUC: {val_auc:.4f} | Val AUC-PR: {val_auc_pr:.4f}")
    
    
    # # Every 20 epochs, calculate and print the optimal threshold
    # if (epoch + 1) % 10 == 0:
    #     optimal_threshold, max_f1,_ = find_optimal_threshold(np.array(all_val_labels), np.array(all_val_predictions))
    #     print(f"Epoch {epoch + 1}: Optimal Threshold: {optimal_threshold}, Max F1 Score: {max_f1:.4f}")

    
    
# Update batch normalization statistics for SWA model
#swa_utils.update_bn(train_loader, swa_model)    
# Save the model's state_dict



# Save the trained encoder
os.makedirs(f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}", exist_ok=True)

model_save_path = f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}"


# # Path to save the model
# swa_model_path = os.path.join(model_save_path, "swa_cross_attention_model.pth")
# #full_model_path = os.path.join(model_save_path, "swa_cross_attention_model_full.pth")

# # Clean state_dict to avoid issues
# swa_state_dict = swa_model.state_dict()
# cleaned_state_dict = {k: v for k, v in swa_state_dict.items() if k != "n_averaged"}

# # Save the cleaned state_dict
# torch.save(cleaned_state_dict, swa_model_path)
# print(f"SWA model state_dict saved to {swa_model_path}")



cross_attention_model_save_path = f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}"


# Path to save the model
cross_attention_model_path = os.path.join(cross_attention_model_save_path, "cross_attention_model.pth")

torch.save(cross_attention_model.state_dict(), cross_attention_model_path )

# # Save the full model (architecture + weights)
# torch.save(swa_model, full_model_path)
# print(f"Full SWA CrossAttentionModel saved to {full_model_path}")







# torch.save(swa_model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")
# Plot Loss Curve
# plot_loss_curve(train_loss_values, val_loss_values, len(train_loss_values))

# # Find Optimal Threshold on Validation Data
# optimal_threshold, max_f1, threshold_metrics = find_optimal_threshold(np.array(all_val_labels), np.array(all_val_predictions))
# print(f"Optimal Threshold: {optimal_threshold}, Max F1 Score: {max_f1}")


# # Save detailed threshold metrics to CSV
# threshold_df = pd.DataFrame(threshold_metrics, columns=["Threshold", "Precision", "Recall", "F1 Score"])
# threshold_df.to_csv(f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/threshold_metrics_{batch_size}_{embedding_dim}_{temperature}.csv", index=False)
# print("Threshold metrics saved to threshold_metrics.csv.")




def evaluate_on_loader(loader, model, optimal_threshold, device, save_path=None, csv_path=None):
    """
    Evaluate metrics on a given DataLoader and optionally save the metrics and predictions to a file.

    Args:
        loader (DataLoader): The DataLoader for evaluation.
        model (nn.Module): The trained model.
        optimal_threshold (float): Optimal threshold for classification.
        device (torch.device): The device for computation.
        save_path (str, optional): Path to save the metrics as a JSON file.
        csv_path (str, optional): Path to save predictions and labels as a CSV file.
    
    Returns:
        dict: Evaluation metrics.
    """
    all_labels, all_predictions = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            # Prepare data
            nucleus_patches = torch.stack([patch.squeeze(0) for patch in batch['nucleus_patches']]).to(device)
            non_nucleus_patches = torch.stack([patch.squeeze(0) for patch in batch['non_nucleus_patches']]).to(device)
            labels = torch.tensor(batch['true_label'], dtype=torch.float).to(device)

            # Forward pass
            outputs, _, _ ,_ = model(nucleus_patches, non_nucleus_patches)
            outputs = outputs.squeeze(dim=-1).cpu().numpy()  # Convert to numpy
            labels = labels.cpu().numpy()

            all_predictions.extend(outputs)
            all_labels.extend(labels)
    
    # Save predictions and labels to a CSV file
    if csv_path:
        results_df = pd.DataFrame({
            "True Label": all_labels,
            "Prediction": all_predictions,
            "Optimal Threshold": [optimal_threshold] * len(all_labels)
        })
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions and labels saved to {csv_path}")
    
    # Calculate metrics using optimal threshold
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions), optimal_threshold)
    
    # Save metrics to file if specified
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {save_path}")
    
    return metrics

# # Evaluate on training, validation, and test sets
# datasets = {
#     "Training": {"loader": train_loader, "save_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/train_metrics.json", "csv_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/train_predictions.csv"},
#     "Validation": {"loader": val_loader, "save_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/val_metrics.json", "csv_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/val_predictions.csv"},
#     "Test": {"loader": test_loader, "save_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/test_metrics.json", "csv_path": f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/test_predictions.csv"}
# }

# # Initialize a dictionary to hold all combined metrics
# combined_metrics = {}

# for dataset_name, info in datasets.items():
#     print(f"Evaluating on {dataset_name} set...")
#     metrics = evaluate_on_loader(
#         loader=info["loader"],
#         model=cross_attention_model,
#         optimal_threshold=optimal_threshold,
#         device=device,
#         save_path=info["save_path"],
#         csv_path=info["csv_path"]
#     )
    
#     # Add the metrics to the combined dictionary under the dataset name
#     combined_metrics[dataset_name] = metrics
    
    
#     for metric_name, metric_value in metrics.items():
#         print(f"{metric_name}: {metric_value:.4f}")
#     print()

# # Save the combined metrics to a single JSON file
# combined_metrics_path = f"./1000epoch_cross_attention_embeddings_overlapping_patches_256dim_{batch_size}_{embedding_dim}_{temperature}_h{hierarchy_weight}/iteration_{iteration}/combined_metrics.json"
# with open(combined_metrics_path, 'w') as f:
#     json.dump(combined_metrics, f, indent=4)
# print(f"Combined metrics saved to {combined_metrics_path}") , this is the attention code, so as per this model you can suggest visualization techniques