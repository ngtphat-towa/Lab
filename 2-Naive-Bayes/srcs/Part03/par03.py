import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    try:
        data = np.loadtxt(filename, delimiter=",", dtype=float)
    except:
        data = np.loadtxt(filename, delimiter=" ", dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def save_combined_plot(accuracies, confusion, dataset_name):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracies, marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy over Epochs - {dataset_name}")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset_name}")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_combined_plot.png")
    plt.close()


def save_results_to_file(accuracy, confusion, dataset_name):
    with open("results.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, confusion, fmt="%d")


def main(trainset_filename, testset_filename, dataset_name):
    # Load train and test data
    X_train, y_train = load_data(trainset_filename)
    X_test, y_test = load_data(testset_filename)

    # Initialize Gaussian Naive Bayes classifier
    clf = GaussianNB()

    # Train classifier
    accuracies = []
    for epoch in range(1, 11):  # Training for 10 epochs
        clf.fit(X_train, y_train)

        # Predict on train set
        y_train_pred = clf.predict(X_train)

        # Calculate training accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        accuracies.append(train_accuracy)

        print(f"Epoch {epoch}: Training Accuracy: {train_accuracy}")

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Final Accuracy on Test Set:", accuracy)

    # Calculate confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion)

    # Save combined plot
    save_combined_plot(accuracies, confusion, dataset_name)
    save_results_to_file(accuracy, confusion, dataset_name)


if __name__ == "__main__":
    datasets = [
        {
            "name": "Iris",
            "train_file": "data//iris//iris.trn",
            "test_file": "data//iris//iris.tst",
        },
        {
            "name": "Optics",
            "train_file": "data//optics//optics.trn",
            "test_file": "data//optics//optics.tst",
        },
        {
            "name": "Letter",
            "train_file": "data//letter//letter.trn",
            "test_file": "data//letter//letter.tst",
        },
        {
            "name": "Leukemia",
            "train_file": "data//leukemia//leukemia.trn",
            "test_file": "data//leukemia//leukemia.tst",
        },
        {
            "name": "Fp",
            "train_file": "data//fp//fp.trn",
            "test_file": "data//fp//fp.tst",
        },
        {
            "name": "Fp017",
            "train_file": "data//fp107//fp107.trn",
            "test_file": "data//fp107//fp107.tst",
        },
    ]

    for dataset in datasets:
        print(f"Dataset: {dataset['name']}")
        trainset_path = os.path.join(dataset["train_file"])
        testset_path = os.path.join(dataset["test_file"])
        main(trainset_path, testset_path, dataset["name"])
        print("\n")
