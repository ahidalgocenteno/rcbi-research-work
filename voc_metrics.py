import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd

def calculate_metrics(csv_path):
    """
    Calculate mAP for multi-class classification.
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing model outputs and ground truth labels
    Returns:
    --------
    dict
        Dictionary containing mAP scores
    """
    df = pd.read_csv(csv_path, header=None)
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    predictions = df.iloc[:, 1:21].values  # Raw prediction scores
    ground_truth = df.iloc[:, 21:41].values
    predictions = 1 / (1 + np.exp(-predictions))
    ground_truth = (ground_truth > 0).astype(int)
    
    num_classes = predictions.shape[1]
    average_precisions = []
    
    for i in range(num_classes):
        ap = average_precision_score(ground_truth[:, i], predictions[:, i])
        average_precisions.append(ap)
    
    mean_ap = np.mean(average_precisions)
    results = {
        'mean_average_precision': mean_ap,
        'per_class_ap': dict(zip(voc_classes, average_precisions))
    }
    return results

def create_results_dataframe(results):
    """
    Create a pandas DataFrame to store the AP metrics
    Parameters:
    -----------
    results : dict
        Dictionary containing mAP scores and per-class AP values
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the results
    """
    # Create a DataFrame with per-class AP values
    df_results = pd.DataFrame([results['per_class_ap']], columns=list(results['per_class_ap'].keys()))
    
    # Add mAP column
    df_results['mAP'] = results['mean_average_precision']
    
    # Add a row name
    df_results.index = ['AP']
    
    return df_results

if __name__ == "__main__":
    csv_path = "code/output_best.csv"
    results = calculate_metrics(csv_path)
    
    # Create and display the results DataFrame
    df_results = create_results_dataframe(results)
    
    # Display the results
    print("\nAverage Precision per Class:")
    print("-" * 120)
    print(df_results.round(4))
    print("-" * 120)
    
    # Export to LaTeX table
    latex_table = df_results.round(4).to_latex(
        float_format="%.4f",
        caption="Average Precision Results per Class",
        label="tab:ap_results",
        escape=False,
        column_format="l" + "c" * len(df_results.columns)  # Left align first column, center align others
    )
    
    # Save the LaTeX table to a file
    with open('ap_results_table_orig.tex', 'w') as f:
        f.write(latex_table)