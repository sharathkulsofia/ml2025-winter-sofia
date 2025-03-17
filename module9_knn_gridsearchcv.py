import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# Function to collect data from the user
def collect_data(set_name):
    while True:
        try:
            num_points = int(input(f"Enter number of points for {set_name} (positive integer): "))
            if num_points > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    X, y = [], []
    for i in range(num_points):
        while True:
            try:
                x = float(input(f"Enter x value for point {i+1}: "))
                y_val = int(input(f"Enter y (class label) for point {i+1} (non-negative integer): "))
                if y_val >= 0:
                    X.append(x)
                    y.append(y_val)
                    break
                else:
                    print("Class label must be a non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter valid numbers.")

    return np.array(X).reshape(-1, 1), np.array(y)

# Collect training data
X_train, y_train = collect_data("Training Set")

# Collect test data
X_test, y_test = collect_data("Test Set")

# Define the kNN classifier
knn = KNeighborsClassifier()

# Define hyperparameter search space for k (from 1 to 10)
param_grid = {'n_neighbors': np.arange(1, 11)}

# Implementing GridSearchCV with cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

# Extract the best k from GridSearch
best_k = grid_search.best_params_['n_neighbors']

# Train the model with the best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Predict on the test set
y_pred = best_knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"\nBest k for kNN Classification: {best_k}")
print(f"Test Accuracy: {accuracy:.2f}")

# Show cross-validation scores for the best k
cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")