import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd

def calculate_map(csv_path):
    """
    Calculate mAP for multi-class classification and store results in a DataFrame.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing model outputs and ground truth labels
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing per-class AP and overall mAP as the last row
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Separate predictions and ground truth
    predictions = df.iloc[:, 1:81].values  # Raw prediction scores
    ground_truth = df.iloc[:, 81:161].values  # Ground truth for 100 classes
    
    # Apply sigmoid to convert logits to probabilities
    predictions = 1 / (1 + np.exp(-predictions))
    
    # Convert ground truth from -1/1 to 0/1
    ground_truth = (ground_truth > 0).astype(int)
    
    # Calculate AP for each class
    num_classes = predictions.shape[1]
    average_precisions = []
    
    for i in range(num_classes):
        # Calculate Average Precision (AP) for this class
        ap = average_precision_score(ground_truth[:, i], predictions[:, i])
        average_precisions.append(ap)
    
    # Calculate mean AP
    mean_ap = np.mean(average_precisions)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Class': list(range(num_classes)) + ['mAP'],
        'Average_Precision': average_precisions + [mean_ap]
    })
    
    return results_df

def display_results(results_df):
    """
    Display the DataFrame with per-class AP and mAP
    """
    print("\nAverage Precision Results:")
    print("-" * 50)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    csv_path = "code/output_best_coco.csv"
    results_df = calculate_map(csv_path)
    display_results(results_df)