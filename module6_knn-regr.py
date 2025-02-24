import numpy as np

class KNNRegression:
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_query):
        # Compute Euclidean (L2) distances
        distances = np.abs(self.X_train - X_query)

        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Compute average Y value of k nearest neighbors
        y_pred = np.mean(self.y_train[k_nearest_indices])
        return y_pred

# Step 1: Read inputs
N = int(input("Enter number of points (N): "))
k = int(input("Enter number of nearest neighbors (k): "))

if k > N:
    print("Error: k cannot be greater than N.")
    exit()

X_train = []
y_train = []

# Step 2: Read N (x, y) points
for i in range(N):
    x = float(input(f"Enter x[{i+1}]: "))
    y = float(input(f"Enter y[{i+1}]: "))
    X_train.append(x)
    y_train.append(y)

# Step 3: Read query X and predict Y using k-NN regression
X_query = float(input("Enter X value to predict Y: "))

# Step 4: Run k-NN Regression
knn_model = KNNRegression(k, X_train, y_train)
y_pred = knn_model.predict(X_query)

# Step 5: Print the result
print(f"Predicted Y value for X = {X_query} is: {y_pred}")