import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
import os
import time
import json

def read_data(file_name):
    return pd.read_csv(file_name)

def manage_single_result(result, path):
    # Compute acuracy
    count_misclassified = (result['y_test'] != result['y_pred']).sum()
    test_accuracy = count_misclassified / len(result['y_test'])

    # Plot confusion matrix
    cm = confusion_matrix(result['y_test'], result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/confusion_matrix.png")

    plt.clf()

    return test_accuracy, result['setting_code'], len(result['doubted_rows'])


def aggregate_results_old(results, path):
    # Take all the settings, sort them by decreasing delta and plot test accuracy
    # wrt the delta

    test_accuracies = []
    deltas = []
    doubts = []
    for result in results.values():
        test_accuracy, setting_code, doubts = manage_single_result(result, path)
        test_accuracies.append(test_accuracy)
        deltas.append(result['delta'])
        doubts.append(doubts)

    # Plot test accuracy wrt delta
    plt.scatter(deltas, test_accuracies)
    plt.xlabel('[DT] - Time to train (s)')
    plt.ylabel('[DT] - Test accuracy')
    plt.title('[DT] - Test accuracy wrt time to train')
    plt.savefig(f"{path}/test_accuracy_vs_time.png")

    plt.clf()

def aggregate_results(results, path):
    # Initialize lists for test accuracies, deltas, doubts, and setting codes
    test_accuracies = []
    deltas = []
    doubts = []
    setting_codes = []

    # Extract data and collect each setting code
    for result in results.values():
        test_accuracy, setting_code, doubt = manage_single_result(result, path)
        test_accuracies.append(test_accuracy)
        deltas.append(result['delta'])
        doubts.append(doubt)
        setting_codes.append(setting_code)

    # Create a colormap to differentiate settings by color
    unique_settings = list(setting_codes) 
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_settings)))
    color_map = {setting: color for setting, color in zip(unique_settings, colors)}

    seen_labels = set()
    for i in range(len(deltas)):
        setting_label = setting_codes[i]
        # Only set the label for this setting if it hasn't been added yet
        if setting_label not in seen_labels:
            plt.scatter(deltas[i], test_accuracies[i], \
                        color=color_map[setting_label], label=setting_label)
            seen_labels.add(setting_label)
        else:
            plt.scatter(deltas[i], test_accuracies[i], color=color_map[setting_label])

    # Add labels, title, and legend to the plot
    plt.xlabel('[DT] - Time to train (s)')
    plt.ylabel('[DT] - Test accuracy')
    plt.title('[DT] - Test accuracy wrt time to train')
    plt.legend(title="Settings")
    plt.savefig(f"{path}/test_accuracy_vs_time.png")
    plt.show()

    plt.clf()

def generate_plots(results, json_path = "./settings/dt_settings.json", \
                    path = "./graphs"):
    '''
    Every dictionary result inside results contains:
    {
        "classifier": tree_classifier,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "doubted_rows": doubted_rows,
        "delta": delta,
        "feature_importances": feature_importances,
        "setting_code": setting_code
    }
    '''
    max_test_accuracy = 0
    best_test_accuracy_setting = None
    min_number_doubts = 1e9
    best_doubts_setting = None

    for result in results.values():
        full_path = os.path.join(path, f"setting_{result['setting_code']}")
        t = manage_single_result(result, full_path)
        if t[0] > max_test_accuracy:
            max_test_accuracy = t[0]
            best_test_accuracy_setting = t[1]
        if t[2] < min_number_doubts:
            min_number_doubts = t[2]
            best_doubts_setting = t[1]

    with open(json_path, "r") as file:
        settings = json.load(file)
    print(f"[DT] - Best test accuracy: {max_test_accuracy} for setting {best_test_accuracy_setting}, which is:")
    print(settings[f"setting_{best_test_accuracy_setting}"])
    print(f"[DT] - Best number of doubts: {min_number_doubts} for setting {best_doubts_setting}, which is:")
    print(settings[f"setting_{best_doubts_setting}"])
    print(f"For this setting, the feature importances are:")
    print(results[best_doubts_setting]['feature_importances'])

    #os.makedirs(os.path.join(path, "DT_best_settings.txt"), exist_ok=True)
    with open(f"{path}/DT_best_settings.txt", "w") as file:
        file.write(f"Best test accuracy: {max_test_accuracy} for setting {best_test_accuracy_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{best_test_accuracy_setting}"], indent=4))
        file.write(f"\nBest number of doubts: {min_number_doubts} for setting {best_doubts_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{best_doubts_setting}"], indent=4))
        file.write(f"\nFor this setting, the feature importances are:\n")
        file.write(results[best_doubts_setting]['feature_importances'].to_string())

    aggregate_results(results, path)


def decision_tree_classification(setting_code, data, features, target, train_input, \
                                train_target, test_input, test_target, \
                                max_depth = None, max_features = None,\
                                random_state=42):
    '''
    Build a decision tree classifier, train it and then test it on the test set.
    Train and test set are obtained by splitting the dataset, for evaluation purposes.

    Assumptions:
        - Default criterion is entropy
        - Default max_depth is None (no maximum depth here)
        - Default random state is 42 for determinism over splits 
            (randomnsess is in feature permutation)

    Parameters:
        - data: pandas DataFrame
            The dataset containing all the features and target
        - features: pandas DataFrame
            The dataset containing all the features
        - target: pandas Series
            The dataset containing the target
        - test_size: float
            The proportion of the dataset to include in the test split
        - random_state: int
            Parameter to build the tree classifier. It allows for deterministic 
            shuffling of features when building the best-splitting tree.
    
    Returns:
        - tree_classifier: DecisionTreeClassifier
            The classifier trained on the training set
        - X_test: pandas DataFrame
            The test set features
        - y_test: pandas Series
            The test set target
        - y_pred: numpy array
            The predicted target values on the
        - doubted_rows: list
            The indices of the rows where the classifier is not sure of 
            the prediction
    '''
    tree_classifier = DecisionTreeClassifier(criterion='entropy',\
                                            max_depth= max_depth,\
                                            max_features=max_features,\
                                            random_state=random_state) # Other defaults are fine
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    X_train, y_train = train_input, train_target
    X_test, y_test = test_input, test_target        # to evaluate 

    start = time.time()
    tree_classifier.fit(X_train, y_train)
    end = time.time()
    delta = end - start

    feature_importances = tree_classifier.feature_importances_
    feature_names = features.columns
    feature_importances = pd.DataFrame(feature_importances, index=feature_names, columns=['importance'])
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    # Let's use predict_proba so that we see the probabilities and choose ourselves
    y_pred = tree_classifier.predict_proba(X_test)
    doubted_rows = []
    for i, row in enumerate(y_pred):
        v1, v2, v3 = row
        differences = [abs(v1 - v2), abs(v1 - v3), abs(v2 - v3)]
        if max(differences) < 0.2:
            doubted_rows.append((X_test.iloc[i], y_test.iloc[i], row))
    
    # print(f"Number of rows with doubt: {len(doubted_rows)}")
    y_pred = np.argmax(y_pred, axis=1)

    performance = tree_classifier.score(X_test, y_test)
    # print(f"[DT] - Accuracy: {performance}")

    '''return tree_classifier, X_test, y_test, y_pred, doubted_rows, delta, \
        feature_importances, setting_code'''
    return_dict = {
        "classifier": tree_classifier,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "doubted_rows": doubted_rows,
        "delta": delta,
        "feature_importances": feature_importances,
        "setting_code": setting_code
    }
    return return_dict

