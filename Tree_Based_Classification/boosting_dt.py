import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
import os, json
import time

ada_setting_keys = ['setting_code', 'n_estimators', 'learning_rate', 'random_state']


def read_data(file_name):
    return pd.read_csv(file_name)

def manage_single_result(result, path, draw = True):
    # Compute acuracy
    count_misclassified = (result['y_test'] != result['y_pred']).sum()
    test_accuracy = count_misclassified / len(result['y_test'])

    if draw:
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
    test_accuracies = []
    deltas = []
    doubts_list = []
    setting_codes = []

    for result in results.values():
        test_accuracy, setting_code, doubts = manage_single_result(result, path, \
                                                                   draw = False)
        test_accuracies.append(test_accuracy)
        deltas.append(result['delay'])
        doubts_list.append(doubts)
        setting_codes.append(setting_code)

    seen_labels = set()
    unique_settings = len(set(setting_codes))
    color_map = plt.cm.get_cmap("tab20c", unique_settings) 
    for i in range(len(deltas)):
        setting_label = setting_codes[i]
        color = color_map(setting_label)
        plt.scatter(deltas[i], test_accuracies[i], color=color)  # Plot each point
        plt.text(deltas[i], test_accuracies[i], str(setting_label), fontsize=8, ha='center', va='center')  # Add the label

    # Add labels, title, and legend to the plot
    plt.xlabel('[ADA] - Time to train (s)')
    plt.ylabel('[ADA] - Test accuracy')
    plt.title('[ADA] - Test accuracy wrt time to train')
    # plt.legend(title="Settings")
    # plt.subplots_adjust(right=0.6)
    # plt.legend(title="Settings", ncol=4, bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{path}/test_accuracy_vs_time.png")
    plt.show()

    plt.clf()


   
def generate_plots(results, json_path, path):
    '''
    Each dictionary result in results will have:
    {
        'classifier': ada_classifier,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'delay': delay,
        'doubted_rows': doubted_rows,
        'feature_importances': feature_importances,
        'setting_code': setting_code
    }
    '''

    max_test_accuracy = 0
    max_test_accuracy_setting = None
    min_number_of_doubts = 1e9
    min_number_of_doubts_setting = None

    for result in results.values():
        full_path = os.path.join(path, f"setting_{result['setting_code']}")
        test_accuracy, setting_code, doubts = manage_single_result(result, full_path)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            max_test_accuracy_setting = setting_code
        if doubts < min_number_of_doubts:
            min_number_of_doubts = doubts
            min_number_of_doubts_setting = setting_code

    with open(json_path, "r") as file:
        settings = json.load(file)
        
    print(f"[ADA] - Best test accuracy: {max_test_accuracy} for setting {max_test_accuracy_setting}, which is:")
    print(settings[f"setting_{max_test_accuracy_setting}"])
    print(f"[ADA] - For this setting, the top 5 feature importances are:")
    print(results[max_test_accuracy_setting]['feature_importances'][:5])

    print(f"[ADA] - Best number of doubts: {min_number_of_doubts} for setting {min_number_of_doubts_setting}, which is:")
    print(settings[f"setting_{min_number_of_doubts_setting}"])
    print(f"[ADA] - For this setting, the top 5 feature importances are:")
    print(results[min_number_of_doubts_setting]['feature_importances'][:5])

    with open(f"{path}/ADA_best_settings.txt", "w") as file:
        file.write(f"Best test accuracy: {max_test_accuracy} for setting {max_test_accuracy_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{max_test_accuracy_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {
            ada_setting_keys[i]: \
                settings[f"setting_{max_test_accuracy_setting}"][i] \
                    for i in range(len(ada_setting_keys))
        }
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"For this setting, the top 5 feature importances are:\n")
        file.write(json.dumps(results[max_test_accuracy_setting]['feature_importances'].to_string(), indent=4))
        
        
        file.write(f"Best number of doubts: {min_number_of_doubts} for setting {min_number_of_doubts_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{min_number_of_doubts_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {
            ada_setting_keys[i]: \
                settings[f"setting_{min_number_of_doubts_setting}"][i] \
                    for i in range(len(ada_setting_keys))
        }
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"For this setting, the top 5 feature importances are:\n")
        file.write(json.dumps(results[min_number_of_doubts_setting]['feature_importances'].to_string(), indent=4))
    aggregate_results(results, path)
    


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

