import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import time
import json

dt_setting_keys = ['setting_code', 'max_depth', 'max_features', 'random_state']

def read_data(file_name):
    return pd.read_csv(file_name)

def get_metrics(class_report):
    weighted_avg = class_report['weighted avg']
    accuracy = class_report['accuracy']
    metrics = (weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1-score'], accuracy) 
    return metrics

def get_best(metrics_1, metrics_2):
    ac1, re1, pr1, f1_1 = metrics_1
    ac2, re2, pr2, f1_2 = metrics_2

    # trivial cases
    if all([ac1 >= ac2, re1 >= re2, pr1 >= pr2, f1_1 >= f1_2]):
        return metrics_1
    elif all([ac1 <= ac2, re1 <= re2, pr1 <= pr2, f1_1 <= f1_2]):
        return metrics_2
    # othereise, solve linear optimization problem giving highest weight to 
    # recall, then f1, then precision and finally accuracy
    else:
        weights = [0.4, 0.3, 0.2, 0.1]
        metrics_1 = [re1, f1_1, pr1, ac1]
        metrics_2 = [re2, f1_2, pr2, ac2]
        return metrics_1 \
            if np.dot(weights, metrics_1) > np.dot(weights, metrics_2) \
            else metrics_2


def manage_single_result(result, path, draw=True):
    # Compute accuracy
    count_misclassified = (result['y_test'] != result['y_pred']).sum()
    count_correct = (result['y_test'] == result['y_pred']).sum()
    test_accuracy = count_correct / len(result['y_test'])
    
    class_report = classification_report(result['y_test'], result['y_pred'], output_dict=True)
    metrics = get_metrics(class_report)
    
    if draw:
        # Set up the figure layout
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio for matrix and text box
        
        # Plot confusion matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax0, cbar=False)
        ax0.set_title('Confusion Matrix')
        ax0.set_xlabel('Predicted')
        ax0.set_ylabel('True')
        
        # Add text box
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.axis('off')  # Hide axes for the text box
        text = "\n".join([
            f"Recall: {metrics[0]:.2f}",
            f"F1-Score: {metrics[1]:.2f}",
            f"Precision: {metrics[2]:.2f}',
            f"Accuracy: {metrics[3]:.2f}"
        ])
        ax1.text(0, 0.5, text, fontsize=10, verticalalignment='center', horizontalalignment='left')
        
        # add legend to the plot (colorbar for the confusion matrix)
        plt.colorbar(ax0.get_children()[0], ax=ax0)

        # Save the figure
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/confusion_matrix_with_text.png")
        plt.clf()
    

    return metrics, test_accuracy, result['setting_code'], len(result['doubted_rows'])


def aggregate_results(results, path):
    test_accuracies = []
    deltas = []
    doubts = []
    setting_codes = []
    test_metrics = []

    for result in results.values():
        metrics, test_accuracy, setting_code, doubt = manage_single_result(result, path, \
                                                                  draw=False)
        test_accuracies.append(test_accuracy)
        test_metrics.append(metrics)
        deltas.append(result['delta'])
        doubts.append(doubt)
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
    plt.xlabel('[DT] - Time to train (s)')
    plt.ylabel('[DT] - Test accuracy')
    plt.title('[DT] - Test accuracy wrt time to train')
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

        model_depth = model.get_depth()

        setting_to_depth[setting_label] = model_depth

    depths = [setting_to_depth[setting] for setting in setting_to_depth]

    color_map = plt.cm.get_cmap("tab20c", unique_settings)
    for i in range(len(depths)):
        setting_label = setting_codes[i]
        color = color_map(setting_label)
        plt.scatter(depths[i], test_accuracies[i], color=color)
        plt.text(depths[i], test_accuracies[i], str(setting_label), fontsize=8, ha='center', va='center')

    plt.xlabel('[DT] - Depth of the tree')
    plt.ylabel('[DT] - Test accuracy')
    plt.title('[DT] - Test accuracy wrt depth of the tree')
    plt.tight_layout()
    plt.savefig(f"{path}/test_accuracy_vs_depth.png")
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
    best_metrics = (0, 0, 0, 0)
    best_metrics_setting = None

    for result in results.values():
        full_path = os.path.join(path, f"setting_{result['setting_code']}")
        t = manage_single_result(result, full_path)
        best_metrics_old = best_metrics
        best_metrics = get_best(best_metrics, t[0])
        if best_metrics != best_metrics_old:
            best_metrics_setting = t[2]
        if t[1] > max_test_accuracy:
            max_test_accuracy = t[1]
            best_test_accuracy_setting = t[2]
        if t[3] < min_number_doubts:
            min_number_doubts = t[3]
            best_doubts_setting = t[2]


    with open(json_path, "r") as file:
        settings = json.load(file)
    print("="*27+" [Decision Tree results] "+"="*28)
    print(f"[DT] - Best test accuracy: {max_test_accuracy} for setting {best_test_accuracy_setting}, which is:")
    print(settings[f"setting_{best_test_accuracy_setting}"])
    print(f"[DT] - For this setting, the top 5 feature importances are:")
    print(results[best_test_accuracy_setting]['feature_importances'][:5])
    
    print(f"[DT] - Best number of doubts: {min_number_doubts} for setting {best_doubts_setting}, which is:")
    print(settings[f"setting_{best_doubts_setting}"])
    print(f"[DT] - For this setting, the top 5feature importances are:")
    print(results[best_doubts_setting]['feature_importances'][:5])

    print(f"[DT] - Best metrics: {best_metrics}, obtained for setting {best_metrics_setting}, which is:")
    print(settings[f"setting_{best_metrics_setting}"])
    print(f"[DT] - For this setting, the top 5 feature importances are:")
    print(results[best_metrics_setting]['feature_importances'][:5])
    
    print("="*80+"\n")
    #os.makedirs(os.path.join(path, "DT_best_settings.txt"), exist_ok=True)
    with open(f"{path}/DT_best_settings.txt", "w") as file:
        file.write(f"Best test accuracy: {max_test_accuracy} for setting {best_test_accuracy_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{best_test_accuracy_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {dt_setting_keys[i]:\
                            settings[f"setting_{best_test_accuracy_setting}"][i] for i in range(len(dt_setting_keys))}
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"\nFor this setting, the feature importances are:\n")
        file.write(results[best_test_accuracy_setting]['feature_importances'].to_string())
        
        
        file.write(f"\nBest number of doubts: {min_number_doubts} for setting {best_doubts_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{best_doubts_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {dt_setting_keys[i]:\
                            settings[f"setting_{best_doubts_setting}"][i] for i in range(len(dt_setting_keys))}
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"\nFor this setting, the feature importances are:\n")
        file.write(results[best_doubts_setting]['feature_importances'].to_string())

        file.write(f"\nBest metrics: {best_metrics}, obtained for setting {best_metrics_setting}, which is:\n")
        file.write(json.dumps(settings[f"setting_{best_metrics_setting}"], indent=4))
        file.write(f"Which corresponds to:\n")
        build_temp_dict = {dt_setting_keys[i]:\
                            settings[f"setting_{best_metrics_setting}"][i] for i in range(len(dt_setting_keys))}
        file.write(json.dumps(build_temp_dict, indent=4))
        file.write(f"\nFor this setting, the feature importances are:\n")
        file.write(results[best_metrics_setting]['feature_importances'].to_string())


    aggregate_results(results, path)

    #plot the tree for the best setting
    result = results[best_metrics_setting]
    fig = plt.figure(figsize=(100,90))
    plot_tree(result['classifier'], filled=True, feature_names=result['X_test'].columns, class_names=['0', '1', '2'])
    # add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.savefig(f"{path}/best_tree.pdf", format="pdf")
    plt.clf()
    return best_test_accuracy_setting


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
    tree_classifier = tree_classifier.fit(X_train, y_train)
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

