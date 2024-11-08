import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
import os, json
import time


rf_setting_keys = ['setting_code', 'n_estimators', 'max_depth', 'max_features', 'random_state', 'bootstrap']


def read_data(file_name):
    return pd.read_csv(file_name)

def manage_single_result(result, path, draw = True):
    # Compute acuracy
    count_misclassified = (result['y_test'] != result['y_pred']).sum()
    count_correct = (result['y_test'] == result['y_pred']).sum()
    test_accuracy = count_correct / len(result['y_test'])

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
        test_accuracy, setting_code, doubts = manage_single_result(result, path,\
                                                                    draw = False)
        test_accuracies.append(test_accuracy)
        deltas.append(result['delta'])
        doubts_list.append(doubts)
        setting_codes.append(setting_code)

    seen_labels = set()
    unique_settings = len(set(setting_codes))
    color_map = plt.cm.get_cmap("tab20c", unique_settings) 
    plt.figure() 
    for i in range(len(deltas)):
        setting_label = setting_codes[i]
        color = color_map(setting_label)
        plt.scatter(deltas[i], test_accuracies[i], color=color)  # Plot each point
        plt.text(deltas[i], test_accuracies[i], str(setting_label), fontsize=8, ha='center', va='center')  # Add the label

    # Add labels, title, and legend to the plot
    plt.xlabel('[RF] - Time to train (s)')
    plt.ylabel('[RF] - Test accuracy')
    # make yticks range from test_accuracy.min() to test_accuracy.max() with step 0.005
    plt.title('[RF] - Test accuracy wrt time to train')
    #plt.legend(title="Settings")
    #plt.subplots_adjust(right=0.6)
    #plt.legend(title="Settings", ncol=4, bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{path}/test_accuracy_vs_time.png")
    #plt.show()

    plt.clf()

    # Plot accuracy wrt tree size
    setting_to_depth = {}
    for result in results.values():
        setting_label = result['setting_code']
        model = result['classifier']

        model_estimators = model.estimators_
        sum = 0
        for e in model_estimators:
            sum += e.get_depth()
        model_depth = sum / len(model_estimators)
        #model_nodes = model.tree_.node_count

        setting_to_depth[setting_label] = model_depth

    depths = [setting_to_depth[setting] for setting in setting_to_depth]

    plt.figure(figsize=(10, 10))
    plt.xlabel('[RF] - Depth of the tree')
    plt.ylabel('[RF] - Test accuracy')
    plt.xlim(0, max(depths) + 5)
    plt.ylim(min(test_accuracies)-0.01, max(test_accuracies)+0.01)
    plt.xticks(range(int(min(depths)), int(max(depths) + 5), 5))
    #plt.yticks(np.arange(min(test_accuracies), 1 + 0.0005, 0.0005))
    plt.yticks(np.arange(min(test_accuracies) - 0.01, max(test_accuracies) + 0.01, 0.002))
    
    ''' FIRST WAY of solving the problem of points being too dense
    # the test_accuracies are still too close for some settings, so:
    test_accuracies = [round(acc, 3) for acc in test_accuracies]
    new_test_accuracies = list(set(test_accuracies))
    new_depths = []
    for i in range(len(new_test_accuracies)):
        new_depths.append(
            (depths[test_accuracies.index(new_test_accuracies[i])],
             setting_codes[test_accuracies.index(new_test_accuracies[i])])
        )
    depths = new_depths
    test_accuracies = new_test_accuracies
    unique_settings = len(set([depths[i][1] for i in range(len(depths))]))
    color_map = plt.cm.get_cmap("tab20c", unique_settings)
    for i in range(len(depths)):
        #setting_label = setting_codes[i]
        setting_label = depths[i][1]
        depth = depths[i][0]
        color = color_map(setting_label)
        plt.scatter(depth, test_accuracies[i], color=color)
        plt.text(depth, test_accuracies[i], str(setting_label), fontsize=8, ha='center', va='center')
    '''

    # SECOND WAY of solving the problem of points being too dense
    depths = [round(depth, 3) for depth in depths]
    new_depths = list(set(depths))
    new_test_accuracies = []
    for i in range(len(new_depths)):
        new_test_accuracies.append(
            (test_accuracies[depths.index(new_depths[i])],
             setting_codes[depths.index(new_depths[i])])
        )
    test_accuracies = new_test_accuracies
    depths = new_depths
    unique_settings = len(set([test_accuracies[i][1] for i in range(len(test_accuracies))]))
    color_map = plt.cm.get_cmap("tab20c", unique_settings)
    for i in range(len(test_accuracies)):
        #setting_label = setting_codes[i]
        setting_label = test_accuracies[i][1]
        test_accuracy = test_accuracies[i][0]
        color = color_map(setting_label)
        plt.scatter(depths[i], test_accuracy, color=color)
        plt.text(depths[i], test_accuracy, str(setting_label), fontsize=8, ha='center', va='center')

    plt.title('[RF] - Test accuracy wrt avg depth of the forest')
    #plt.tight_layout()
    plt.savefig(f"{path}/test_accuracy_vs_depth.png")
    plt.clf()

   
def generate_plots(results, json_path, path):
    '''
    Every dictionary 'result' inside 'results' has the following keys:
    {
        "classifier": rf_classifier,
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
    max_test_accuracy_setting = None
    min_number_doubts = 1e9
    min_number_doubts_setting = None

    for result in results.values():
        full_path = os.path.join(path, f"setting_{result['setting_code']}")
        test_accuracy, setting_code, doubts = manage_single_result(result, full_path)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            max_test_accuracy_setting = setting_code
        if doubts < min_number_doubts:
            min_number_doubts = doubts
            min_number_doubts_setting = setting_code

    with open(json_path, 'r') as json_file:
        settings = json.load(json_file)

    print("="*27 + " [Random Forest Results] " + "="*28)
    print(f"[RF] - Max test accuracy: {max_test_accuracy} for setting {max_test_accuracy_setting}, which is:")
    print(settings[f"setting_{max_test_accuracy_setting}"])
    print(f"[RF] - For setting {max_test_accuracy_setting}, the top 5 feature importances are:")
    print(results[max_test_accuracy_setting]['feature_importances'][:5])

    print(f"[RF] - Min number of doubts: {min_number_doubts} for setting {min_number_doubts_setting}, which is:")
    print(settings[f"setting_{min_number_doubts_setting}"])
    print(f"[RF] - For setting {min_number_doubts_setting}, the top 5 feature importances are:")
    print(results[min_number_doubts_setting]['feature_importances'][:5])

    print("="*80 + "\n")

    with open(f"{path}/RF_best_settings.json", "w") as file:
        file.write(f"Best test accuracy: {max_test_accuracy} for setting {max_test_accuracy_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{max_test_accuracy_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {
            rf_setting_keys[i]:\
                settings[f"setting_{max_test_accuracy_setting}"][i]\
                for i in range(len(rf_setting_keys))
        }
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"\nFor this setting, the top 5 feature importances are:\n")
        file.write(str(results[max_test_accuracy_setting]['feature_importances'].to_string()))

        file.write(f"\n\nMin number of doubts: {min_number_doubts} for setting {min_number_doubts_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{min_number_doubts_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {
            rf_setting_keys[i]:\
                settings[f"setting_{min_number_doubts_setting}"][i]\
                for i in range(len(rf_setting_keys))
        }
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"\nFor this setting, the top 5 feature importances are:\n")
        file.write(str(results[min_number_doubts_setting]['feature_importances'].to_string()))

    aggregate_results(results, path)
    return max_test_accuracy_setting



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
    rf_classifier = rf_classifier.fit(X_train, y_train)
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


