import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def main():
    # Ask the user for number of points and k (neighbors)
    N = int(input("Enter number of points (N): "))
    k = int(input("Enter number of nearest neighbors (k): "))

    if k > N:
        print("Error: k cannot be greater than N.")
        return

    # Read N (x, y) points
    x_values = []
    y_values = []
    for i in range(N):
        x_val = float(input(f"Enter x value for point {i+1}: "))
        y_val = float(input(f"Enter y value for point {i+1}: "))
        x_values.append(x_val)
        y_values.append(y_val)

    # Convert lists to numpy arrays (X needs to be 2D for scikit-learn)
    X_train = np.array(x_values).reshape(-1, 1)
    y_train = np.array(y_values)

    # Read query X
    query = float(input("Enter the query X value to predict Y: "))

    # Create and train k-NN regression model using scikit-learn
    knn_regressor = KNeighborsRegressor(n_neighbors=k, metric="euclidean")
    knn_regressor.fit(X_train, y_train)

    # Predict the Y value for the given query X
    prediction = knn_regressor.predict(np.array([[query]]))

    # Compute variance of training labels using NumPy
    variance = np.var(y_train)

    # Output the predicted Y and the variance
    print(f"Predicted Y value for X = {query}: {prediction[0]}")
    print(f"Variance of training labels: {variance}")

if __name__ == '__main__':
    main()