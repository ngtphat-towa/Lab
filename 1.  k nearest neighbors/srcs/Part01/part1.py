import math
import numpy as np
import csv

# Load the training data
with open("training_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    train_data = np.array(list(reader))

# Load the test data
with open("test_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    test_data = np.array(list(reader))

# Separate features and labels
train_features, train_labels = train_data[:, :-1], train_data[:, -1]
test_features = test_data[:, :-1]

# Print the loaded data
print("Training data features:\n", train_features)
print("Training data labels:\n", train_labels)
print("Test data features:\n", test_features)

k_list = [1, 3]
header = ["X1", "X2", "Class"]


def euclidean_distance(point1, point2):
    squared_distances = 0
    for i in range(len(point1)):
        squared_distances += (float(point1[i]) - float(point2[i])) ** 2
    return math.sqrt(squared_distances)


def get_neighbors(train_features, train_labels, test_point, k):
    distances = []
    for i in range(len(train_features)):
        distances.append(
            (euclidean_distance(train_features[i], test_point), train_labels[i])
        )
    distances.sort(key=lambda x: x[0])
    return [label for distance, label in distances[:k]]


def knn_predict(train_features, train_labels, test_features, k):
    predictions = []
    for test_point in test_features:
        neighbors = get_neighbors(train_features, train_labels, test_point, k)
        class_counts = {}
        for label in neighbors:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        majority_vote = max(class_counts, key=class_counts.get)  # type: ignore
        predictions.append(majority_vote)
    return predictions


for k in k_list:
    predictions = knn_predict(train_features, train_labels, test_features, k)
    print(f"Predictions for k = {k}:\n", predictions)
