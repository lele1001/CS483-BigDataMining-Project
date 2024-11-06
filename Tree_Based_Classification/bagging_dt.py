import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
import os
import time


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

def aggregate_results(results, path):
    # Initialize lists for test accuracies, deltas, doubts, and setting codes
    test_accuracies = []
    deltas = []
    doubts = []
    setting_codes = []

    # Extract data and collect each setting code
    for result in results.values():
        test_accuracy, setting_code, doubts = manage_single_result(result, path)
        test_accuracies.append(test_accuracy)
        deltas.append(result['delta'])
        doubts.append(doubts)
        setting_codes.append(setting_code)

   
def generate_plots(results, json_path, path):
    # TODO
    print("Not implemented yet")
    pass

def random_forest_classification(setting_code, data, features, target,  train_input, \
                                train_target, test_input, test_target, \
                                n_estimators =  100, max_depth = None, \
                                max_features = 'sqrt', random_state=42, \
                                n_jobs= -1, bootstrap = False):
    '''
    Build a random forest classifier, train it and then test it on the test set.
    Train and test set are obtained by splitting the dataset, for evaluation purposes.

    Assumptions:
        - Default criterion is entropy
        - Default max_depth is None (no maximum depth here)
        - Default random state is 42 for determinism over splits
            (randomnsess is in feature permutation)
        - Default n_estimators is 100
        - Default max_features is None (all features are considered)
        - We use all available cores for parallel processing by default
        - Feature bagging, with whole dataset used for each tree (bootstrap=False)
        - Max features for each tree is sqrt(n_features) (empirical result, see documentation)

    Parameters:

        - data: pandas DataFrame
            The dataset containing all the features and target
        - features: pandas DataFrame
            The dataset containing all the features
        - target: pandas Series
            The dataset containing the target
        - n_estimators: int
            The number of trees in the forest
        - max_depth: int
            The maximum depth of the tree
        - max_features: int
            The number of features to consider when looking for the best split
        - test_size: float
            The proportion of the dataset to include in the test split
        - random_state: int
            Parameter to build the tree classifier. It allows for deterministic 
            shuffling of features when building the best-splitting tree.
        - n_jobs: int
            The number of jobs to run in parallel for both fit and predict. 
            If -1, then the number of jobs is set to the number of cores.

    Returns:
        - rf_classifier: RandomForestClassifier
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
        - delta: float
            The time taken to train the model
    '''

    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    X_train, y_train = train_input, train_target
    X_test, y_test = test_input, test_target


    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, \
                                        max_depth=max_depth,\
                                        max_features=max_features,\
                                        random_state=random_state, n_jobs=n_jobs, \
                                        bootstrap=bootstrap)
    start_time = time.time()
    rf_classifier.fit(X_train, y_train)
    end_time = time.time()
    delta = end_time - start_time  

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_
    feature_importances = pd.Series(feature_importances, index=features.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_importances = pd.DataFrame(feature_importances, columns=['importance'])


    y_pred = rf_classifier.predict_proba(X_test)

    doubted_rows = []
    for i in range(len(y_pred)):
        v1, v2, v3 = y_pred[i]
        differences = [abs(v1 - v2), abs(v1 - v3), abs(v2 - v3)]
        if max(differences) < 0.2:
            doubted_rows.append((X_test.iloc[i], y_test.iloc[i], y_pred[i]))

    y_pred = np.argmax(y_pred, axis=1)
    # print(f"Number of rows with doubt: {len(doubted_rows)}")
    # print(f"[Random Forest] - Accuracy: {accuracy_score(y_test, y_pred)}")

    '''return rf_classifier, X_test, y_test, y_pred, doubted_rows, delta, \
        feature_importances, setting_code'''
    
    return_dict = {
        "classifier": rf_classifier,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "doubted_rows": doubted_rows,
        "delta": delta,
        "feature_importances": feature_importances,
        "setting_code": setting_code
    }
    return return_dict


