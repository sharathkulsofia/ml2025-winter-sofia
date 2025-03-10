import numpy as np
from sklearn.metrics import precision_score, recall_score

def main():
    # Ask user for the number of points
    N = int(input("Enter the number of points (N): "))
    
    # Initialize lists to store ground truth (x) and predicted (y) values
    ground_truth = []
    predictions = []
    
    # Read N (x, y) points one by one
    # x is the ground truth, y is the predicted class label
    for i in range(N):
        x_val = int(input(f"Enter ground truth class (0 or 1) for point {i+1}: "))
        y_val = int(input(f"Enter predicted class (0 or 1) for point {i+1}: "))
        ground_truth.append(x_val)
        predictions.append(y_val)
    
    # Convert lists to NumPy arrays for further processing
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    # Use scikit-learn to compute Precision and Recall
    # (Assuming class '1' is the positive label)
    precision = precision_score(ground_truth, predictions, pos_label=1)
    recall = recall_score(ground_truth, predictions, pos_label=1)
    
    # Print the results rounded to two decimal places
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))

if __name__ == '__main__':
    main()