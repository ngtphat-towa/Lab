import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filename):
    """Loads data from a CSV or space-delimited text file."""
    try:
        data = np.loadtxt(filename, delimiter=",", dtype=float)
    except:
        data = np.loadtxt(filename, delimiter=" ", dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def print_confusion_matrix(confusion):
    """Prints the confusion matrix in text format."""
    for row in confusion:
        print(row)


def save_results_to_file(accuracy, confusion, features, dataset_name):
    """Saves accuracy, confusion matrix, and tree features to a text file named "results.txt"."""
    with open("results.txt", "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nTree Features:\n")
        f.write(features)
        f.write("Confusion Matrix:\n")
        np.savetxt(f, confusion, fmt="%d")


def export_confusion_matrix(confusion, accuracy, num_nodes, dataset_name):
    """Generates and saves a confusion matrix heatmap image."""
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        f"Confusion Matrix - {dataset_name} (Accuracy: {accuracy:.4f}, Nodes: {num_nodes})"
    )

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_combined_plot.png")
    plt.close()


def test_model(clf, X_test, y_test):
    """Makes predictions using the classifier, calculates accuracy, and returns both."""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def get_tree_features_context(clf):
    """Extracts relevant features from the trained decision tree."""
    num_nodes = clf.tree_.node_count
    max_depth = clf.tree_.max_depth
    num_features = clf.n_classes_
    features = (
        f"Number of Nodes: {num_nodes}\n"
        f"Maximum Depth: {max_depth}\n"
        f"Number of Features: {num_features}\n"
    )
    return features


def decision_tree_classification(trainset_filename, testset_filename, dataset_name=""):
    """Performs decision tree classification for a single dataset."""
    # Load train and test data
    X_train, y_train = load_data(trainset_filename)
    X_test, y_test = load_data(testset_filename)

    # Initialize DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    # Train classifier
    clf.fit(X_train, y_train)

    # Get tree features
    features = get_tree_features_context(clf)

    # Test and evaluate
    accuracy, y_pred = test_model(clf, X_test, y_test)
    confusion = confusion_matrix(y_test, y_pred)

    # Print the result to console
    print(f"\nDataset: {dataset_name}")
    print("Test Accuracy:", accuracy)
    print(features)
    print("\nConfusion Matrix:")
    print_confusion_matrix(confusion)

    # Save results
    num_nodes = clf.tree_.node_count
    export_confusion_matrix(confusion, accuracy, num_nodes, dataset_name)
    save_results_to_file(accuracy, confusion, features, dataset_name)


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
        decision_tree_classification(
            dataset["train_file"], dataset["test_file"], dataset["name"]
        )
        print("\n")
