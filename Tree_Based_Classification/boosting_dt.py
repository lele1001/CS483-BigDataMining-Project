import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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


def AdaBoost_classification(setting_code, data, features, target,  train_input, \
                            train_target, test_input, test_target, \
                            n_estimators =  100,learning_rate = 1,\
                            random_state=42):
    '''
    Build an AdaBoost classifier, train it and then test it on the test set.
    Train and test set are obtained by splitting the dataset, for evaluation purposes.

    Assumptions:
        - Default base estimator is a decision tree
        - Default learning rate is 1
        - Default random state is 42 for determinism over splits
            (randomnsess is in feature permutation)
        - Default n_estimators is 100

    Parameters:
        - data: pandas DataFrame
            The dataset containing all the features and target
        - features: pandas DataFrame
            The dataset containing all the features
        - target: pandas Series
            The dataset containing the target
        - n_estimators: int
            The maximum number of estimators at which boosting is terminated
        - learning_rate: float
            The contribution of each classifier to the weights
        - test_size: float
            The proportion of the dataset to include in the test split
        - random_state: int
            Parameter to build the tree classifier. It allows for deterministic 
            shuffling of features when building the best-splitting tree.
    
    Returns:
        - ada_classifier: AdaBoostClassifier
            The classifier trained on the training set
        - X_test: pandas DataFrame
            The test set features
        - y_test: pandas Series
            The test set target
    '''
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

    X_train, y_train = train_input, train_target
    X_test, y_test = test_input, test_target


    ada_classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, algorithm='SAMME')
    start = time.time()
    ada_classifier.fit(X_train, y_train)
    end = time.time()
    delay = end - start

    # Get feature importances
    feature_importances = ada_classifier.feature_importances_
    feature_importances = pd.Series(feature_importances, index=features.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    feature_importances = pd.DataFrame(feature_importances, columns=['importance'])

    y_pred = ada_classifier.predict_proba(X_test)
    doubted_rows = []
    for i in range(len(y_pred)):
        v1, v2, v3 = y_pred[i]
        differences = [abs(v1 - v2), abs(v1 - v3), abs(v2 - v3)]
        if max(differences) < 0.2:
            doubted_rows.append((X_test.iloc[i], y_test.iloc[i], y_pred[i]))

    y_pred = np.argmax(y_pred, axis=1)

    '''return ada_classifier, X_test, y_test, y_pred, delay, doubted_rows, \
            feature_importances, setting_code'''
    
    return_dict = {
        'classifier': ada_classifier,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'delay': delay,
        'doubted_rows': doubted_rows,
        'feature_importances': feature_importances,
        'setting_code': setting_code
    }
    return return_dict

