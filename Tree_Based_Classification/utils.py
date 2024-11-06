import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bagging_dt, boosting_dt, decision_trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os, shutil
#import regex as re
from scipy.linalg import svd as SVD
import json

def read_data(file_name):
    return pd.read_csv(file_name)

def clean_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  

def describe_settings(dt_settings, rf_settings, ada_settings):
    # create json files for each of the 3 of settings, dt, rf and ada
    clean_directory("./settings")
    dt_settings_dict = {}
    rf_settings_dict = {}
    ada_settings_dict = {}

    print(f"Describing settings...")
    print(f"\tlen(dt_settings): {len(dt_settings)}")
    print(f"\tlen(rf_settings): {len(rf_settings)}")
    print(f"\tlen(ada_settings): {len(ada_settings)}\n")

    for i, dt_setting in enumerate(dt_settings):
        dt_settings_dict[f"setting_{i}"] = dt_setting
        #print(f"\tIterating over dt_settings...\n")
    with open("./settings/dt_settings.json", "w") as file:
        file.write(json.dumps(dt_settings_dict, indent=4))

    for i, rf_setting in enumerate(rf_settings):
        rf_settings_dict[f"setting_{i}"] = rf_setting
        #print(f"\tIterating over rf_settings...\n")
    with open("./settings/rf_settings.json", "w") as file:
        file.write(json.dumps(rf_settings_dict, indent=4))

    for i, ada_setting in enumerate(ada_settings):
        ada_settings_dict[f"setting_{i}"] = ada_setting
        #print(f"\tIterating over ada_settings...\n")
    with open("./settings/ada_settings.json", "w") as file:
        #print(f"Writing AdaBoost settings to file")
        #print(ada_settings_dict)
        file.write(json.dumps(ada_settings_dict, indent=4))

def select_features(only_features_df, number_of_features):
    # Use SVG to know which is the number of most significant features
    U, Sigma, V_t = SVD(only_features_df)
    Sigma_sq = Sigma**2
    target_percentage_tries = [0.9, 0.85, 0.80, 0.75]
    dict = {}
    total_variance = np.sum(Sigma_sq)
    for target_percentage in target_percentage_tries:
        current_variance = 0
        for i in range(number_of_features):
            current_variance += Sigma_sq[i]
            if current_variance/total_variance >= target_percentage:
                number_of_most_significant_features = i
                break
        dict[target_percentage] = number_of_most_significant_features
        print(f"Number of most significant features for {target_percentage} target percentage: {number_of_most_significant_features}")

    # As target_percentage decreases, the number of most significant features decreases too
    # let's choose the one with largest difference from the previous one
    differences = []
    for i in range(1, len(target_percentage_tries)):
        differences.append(dict[target_percentage_tries[i]] - dict[target_percentage_tries[i-1]])
    max_diff = max(differences)
    target_percentage = target_percentage_tries[differences.index(max_diff) + 1]
    number_of_most_significant_features = dict[target_percentage]
    print(f"Chosen target percentage: {target_percentage}," 
         + f"implying a number of most significant features of {number_of_most_significant_features}")
    return target_percentage, number_of_most_significant_features

def make_dirs(root, subdirs):
    if not os.path.exists(root):
        os.makedirs(root)
    else:
        for subdir in subdirs:
            if not os.path.exists(f"{root}/{subdir}"):
                os.makedirs(f"{root}/{subdir}")
        

def make_plots(dt_results, rf_results, ada_results):
    subdirs = ["Random_Forest", "AdaBoost", "Decision_Tree"]
    clean_directory("./graphs")
    make_dirs("./graphs", subdirs)

    '''if rf_results['classifier'].isinstance(RandomForestClassifier):
        bagging_dt.generate_plots(rf_results, os.path.join("./graphs", "Random_Forest"))
    elif ada_results['classifier'].isinstance(AdaBoostClassifier):
        boosting_dt.generate_plots(ada_results, os.path.join("./graphs", "AdaBoost"))
    elif dt_results['classifier'].isinstance(DecisionTreeClassifier):
        decision_trees.generate_plots(dt_results, os.path.join("./graphs", "Decision_Tree"))
    else:
        raise ValueError("Classifier not recognized")'''
    bagging_dt.generate_plots(rf_results, json_path = "./settings/dt_settings.json", \
                              path =  os.path.join("./graphs", "Random_Forest"))
    print(f"Finished generating plots for Random Forest")
    boosting_dt.generate_plots(ada_results, json_path = "./settings/dt_settings.json",\
                                path = os.path.join("./graphs", "AdaBoost"))
    print(f"Finished generating plots for AdaBoost")
    decision_trees.generate_plots(dt_results,json_path = "./settings/dt_settings.json",\
                                   path = os.path.join("./graphs", "Decision_Tree"))
    print(f"Finished generating plots for Decision Tree")


        