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


def export_confusion_matrix(confusion, dataset_name):
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset_name}")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_combined_plot.png")
    plt.close()


def print_confusion_matrix(confusion):
    for row in confusion:
        print(row)


def save_results_to_file(accuracy, confusion, dataset_name):
    with open("results.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, confusion, fmt="%d")


def test_model(clf, X_test, y_test):

    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred


def main(trainset_filename, testset_filename, dataset_name=""):
    # Load train and test data
    X_train, y_train = load_data(trainset_filename)
    X_test, y_test = load_data(testset_filename)

    # Initialize Gaussian Naive Bayes classifier
    clf = GaussianNB()

    # Train classifier
    clf.fit(X_train, y_train)

    # Predict on test
    train_accuracy, _ = test_model(clf, X_train, y_train)
    accuracy, y_pred = test_model(clf, X_test, y_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", accuracy)

    # Calculate confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print_confusion_matrix(confusion)

    # Save into file
    export_confusion_matrix(confusion, dataset_name)
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
