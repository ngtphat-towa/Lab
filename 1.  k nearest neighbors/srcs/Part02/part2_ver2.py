import time
import numpy as np

# Function to calculate Euclidean distance between two sets of points
def euclidean_distance(test, train):
    return np.sqrt(np.sum(np.square(test - train), axis=1))

# Function to perform k-nearest neighbors classification
def kNN(trainset_values, trainset_labels, testset_values, k, metric=euclidean_distance):
    testset_predictions = []
    for test_value in testset_values:
        distances = metric(test_value, trainset_values)
        indices = np.argsort(distances)[:k]
        neighbors = trainset_labels[indices]
        testset_predictions.append(np.argmax(np.bincount(neighbors)))
    return testset_predictions

# Function to create a confusion matrix from predictions and true labels
def create_confusion_matrix(prediction, original):
    labels = np.unique(original)
    amount = len(labels)
    confusion_matrix = np.zeros((amount, amount))
    # Fill in the confusion matrix with the number of predictions for each pair of true and predicted labels
    for i in range(amount):
        for j in range(amount):
            confusion_matrix[i, j] = np.sum((original == labels[i]) & (prediction == labels[j]))
    return confusion_matrix.astype(int)

# Function to calculate classification accuracy
def calculate_accuracy(testset_labels, testset_predictions):
    correct = 0
    for i in range(len(testset_labels)):
        if (testset_labels[i] == testset_predictions[i]):
            correct += 1
    accuracy = correct / len(testset_labels)
    return accuracy

# List of datasets containing training and testing data
list_datasets = [
    {'train_file': 'data//iris//iris.trn', 'test_file': 'data//iris//iris.tst'}, 
    {'train_file': 'data//optics//optics.trn', 'test_file': 'data//optics//optics.tst'}, 
    {'train_file': 'data//letter//letter.trn', 'test_file': 'data//letter//letter.tst'}, 
    {'train_file': 'data//faces//faces.trn', 'test_file': 'data//faces//faces.tst'}, 
    {'train_file': 'data//fp107//fp107.trn', 'test_file': 'data//fp107//fp107.tst'}
]


# List of k values to try
k_list = [1, 5, 7]


output_file = "kNN_results.txt"

# Open the file for writing
with open(output_file, 'w') as f_out:
    # Loop over all datasets and k values
    for dataset in list_datasets:
        try:
            # Load training and testing data from files
            train_data = np.loadtxt(dataset['train_file'], delimiter=',', dtype=float)
            test_data = np.loadtxt(dataset['test_file'], delimiter=',', dtype=float)
        except:
            train_data = np.loadtxt(dataset['train_file'], delimiter=' ', dtype=float)
            test_data = np.loadtxt(dataset['test_file'], delimiter=' ', dtype=float)
        
        # Split training and testing data into values and labels
        trainset_values, trainset_labels = train_data[:, :-1], train_data[:, -1].astype(int)
        testset_values, testset_labels = test_data[:, :-1], test_data[:, -1].astype(int)
        
        for k_value in k_list:
            start_time = time.time()
            
            # Call kNN function and store the predictions
            predictions = kNN(trainset_values, trainset_labels, testset_values, k_value)
        
            # Calculate accuracy and create confusion matrix
            accuracy = calculate_accuracy(testset_labels, predictions)
            confusion_matrix = create_confusion_matrix(predictions, testset_labels)

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Print the results to the console
            print(f"\nDataset: {dataset}\n k = {k_value}")
            print(f"Accuracy: {accuracy}")
            print(f'Confusion Matrix:\n{confusion_matrix}')
            print(f"Elapsed Time: {elapsed_time:.4f} seconds")
            
            # Write the results to the output file
            f_out.write(f"\nDataset: {dataset}\n k = {k_value}\n")
            f_out.write(f"Accuracy: {accuracy}\n")
            f_out.write(f'Confusion Matrix: {len(confusion_matrix)}\n')
            f_out.write(f"{confusion_matrix}")
            f_out.write(f"\nElapsed Time: {elapsed_time:.4f} seconds\n")

print(f"\nResults have been written to {output_file}")