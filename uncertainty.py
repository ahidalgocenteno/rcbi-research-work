import os
import torch
import argparse
from tqdm import tqdm
from models import get_model
from data import make_data_loader

# Argument parser setup
parser = argparse.ArgumentParser(description='PyTorch Multi-label Image Classification with Uncertainty')

# General settings
parser.add_argument('--data_root_dir', default='./dataset/', type=str, help='Root directory for dataset')
parser.add_argument('--image-size', '-i', default=448, type=int, help='Image size for training')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_samples', default=50, type=int, help='Number of stochastic forward passes for MC Dropout')
parser.add_argument('--num-workers', default=4, type=int, metavar='INT', help='Number of data loading workers (default: 4)')
parser.add_argument('--word-feature-path', default='./wordfeature/', type=str, help='Path to word features')
parser.add_argument('--data', default='VOC2007', type=str, help='Dataset name (e.g., VOC2007, COCO)')
parser.add_argument('--model-name', default='MLGCNDropout', type=str, help='Model name')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model on validation set')
parser.add_argument('--test-efficient', action='store_true', help='Test efficient model')

# Parse arguments
args = parser.parse_args()

def main(args):
    # Use the arguments for device and dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    # Data loaders
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=False)

    # Initialize the model
    model = get_model(num_classes, args)
    model.to(device)

    # Enable MC Dropout
    model.activate_mc_dropout()

    # Test function to compute uncertainty
    def test_with_uncertainty(model, val_loader, num_samples):
        model.eval()  # Set the model to evaluation mode
        all_means = []
        all_uncertainties = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Testing Uncertainty"):
                images, labels = batch["image"].to(device), batch["target"].to(device)
                outputs = []

                # Perform multiple stochastic forward passes
                for _ in range(num_samples):
                    outputs.append(torch.sigmoid(model(images)))  # Apply sigmoid for probabilities

                # Stack outputs to calculate mean and uncertainty
                outputs = torch.stack(outputs, dim=0)  # [num_samples, batch_size, num_classes]
                mean = outputs.mean(dim=0)  # Mean prediction
                uncertainty = outputs.std(dim=0)  # Uncertainty (standard deviation)

                all_means.append(mean)
                all_uncertainties.append(uncertainty)

        return torch.cat(all_means), torch.cat(all_uncertainties)

    # Compute predictions and uncertainty
    mean_preds, uncertainty = test_with_uncertainty(model, val_loader, args.num_samples)
    print("Mean Predictions:", mean_preds)
    print("Uncertainty:", uncertainty)

if __name__ == "__main__":
    main(args)
