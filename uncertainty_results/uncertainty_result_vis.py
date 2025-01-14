import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='Uncertainty Visualization')

# get the dataset name
parser.add_argument('--data', type=str, default='VOC2007', help='Dataset name')
parser.add_argument('--uncertainty_method', type=str, default='taa', help='Uncertainty method')

def vis_uncertainty_results(variances, entropies, label_names):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot variances
    sns.barplot(x=label_names, y=variances, ax=axes[0])
    axes[0].set_title('Variances per Label')
    axes[0].set_xlabel('Labels')
    axes[0].set_ylabel('Variance')
    axes[0].tick_params(axis='x', rotation=90)

    # Plot entropies
    sns.barplot(x=label_names, y=entropies, ax=axes[1])
    axes[1].set_title('Entropies per Label')
    axes[1].set_xlabel('Labels')
    axes[1].set_ylabel('Entropy')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create an argument parser
    args = parser.parse_args()
    if args.data == 'VOC2007':
        labels_dict = {
            0: "aeroplane",
            1: "bicycle",
            2: "bird",
            3: "boat",
            4: "bottle",
            5: "bus",
            6: "car",
            7: "cat",
            8: "chair",
            9: "cow",
            10: "diningtable",
            11: "dog",
            12: "horse",
            13: "motorbike",
            14: "person",
            15: "pottedplant",
            16: "sheep",
            17: "sofa",
            18: "train",
            19: "tvmonitor"
        }
    else:
        labels_dict = {}

    # Create a list of label names in the order of class indices
    label_names = [labels_dict[i] for i in range(len(labels_dict))]

    file = os.path.join("uncertainty_results", args.data + "_per_label_uncertainty_" + args.uncertainty_method + ".pkl")

    # Load the pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)

    variances = np.array(data["variances"])
    entropies = np.array(data["entropies"])

    # Visualize the uncertainty results
    vis_uncertainty_results(variances, entropies, label_names)


