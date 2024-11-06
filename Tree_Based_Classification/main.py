import utils
import bagging_dt
import boosting_dt
import decision_trees

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd as SVD
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    # Read data
    input_file_path = "../data/labels.csv"
    #data = utils.read_data(input_file_path)
    data  = pd.read_csv(input_file_path)
    only_features_df = data.drop(columns=['Diabetes_012'])
    target_df = data['Diabetes_012']

    number_of_features = len(only_features_df.columns)
    number_of_most_significant_features = number_of_features
    
    # target_percentage, number_of_most_significant_features = \
    #     utils.select_features(only_features_df, number_of_features)

    # Get ready for parameter exploration
    concat = [number_of_most_significant_features] \
                if number_of_most_significant_features < number_of_features \
                else []
    dt_parameters = {
        'max_depth': [None,9,5,3],
        'max_features': ['sqrt', number_of_features] + concat, # sqrt is default
        'random_state': [42]      # Default for now
    }

    rf_parameters = {
        'n_estimators': [100, 50, number_of_features, 10],
        'max_depth': [None, 9, 5, 3],
        'max_features': ['sqrt', number_of_features]+concat,    # sqrt is default
        'random_state': [42],             # Default for now
        'n_jobs': -1,                   # All available cores
        'bootstrap': [False, True]      # Instance bagging off, then on
    }

    ada_parameters = {
        'n_estimators': [100, 50, 10],
        'learning_rate': [1, 0.5, 0.1],
        'random_state': [42]              # Default for now
    }

    # Decision Tree and AdaBoost can be run in parallel, whereas Random Forest 
    # needs all the cores for itslef

    number_of_cores = os.cpu_count()
    print(f"Number of cores: {number_of_cores}")

    print(f"Splitting dataset in train and test...")
    # Split dataset in train and test
    train_input, test_input, train_target, test_target = \
        train_test_split(only_features_df, target_df, test_size=0.2, random_state=42)

    print(f"Starting the exploration of the parameters space...")
    dt_settings_queue = []
    dt_settings_queue_slim = []
    i = 0
    for random_state in dt_parameters['random_state']:
        for max_depth in dt_parameters['max_depth']:
            for max_features in dt_parameters['max_features']:
                dt_settings_queue.append((i, data, only_features_df, target_df, \
                    train_input, train_target, test_input, test_target, \
                    max_depth, max_features, random_state))
                dt_settings_queue_slim.append((i, max_depth, max_features,\
                                                random_state))
                i += 1 
    rf_settings_queue = []
    rf_settings_queue_slim = []
    i = 0
    for random_state in rf_parameters['random_state']:
        for n_estimators in rf_parameters['n_estimators']:
            for max_depth in rf_parameters['max_depth']:
                for max_features in rf_parameters['max_features']:
                    for bootstrap in rf_parameters['bootstrap']:
                        rf_settings_queue.append((i, data, only_features_df, target_df, \
                            train_input, train_target, test_input, test_target, \
                            n_estimators, max_depth, max_features, random_state, -1, bootstrap))
                        rf_settings_queue_slim.append((i, n_estimators, max_depth, max_features,\
                                                        random_state, bootstrap))
                        i += 1
                        
    ada_settings_queue = []
    ada_settings_queue_slim = []
    i = 0
    for random_state in ada_parameters['random_state']:
        for n_estimators in ada_parameters['n_estimators']:
            for learning_rate in ada_parameters['learning_rate']:
                ada_settings_queue.append((i, data, only_features_df, target_df, \
                    train_input, train_target, test_input, test_target, \
                    n_estimators, learning_rate, random_state))
                ada_settings_queue_slim.append((i, n_estimators, learning_rate, random_state))
                i += 1

    print(f"Saving settings to file...")
    utils.describe_settings(dt_settings_queue_slim, rf_settings_queue_slim, ada_settings_queue_slim)
    
    '''            
    Fill the available cores with Decision Trees and AdaBoost first
    Then, Random Forest will take all the cores
    assign tasks to cores
    '''

    print(f"Start running DT and ADA in parallel...")
    # run subprocesses in parallel
    with ProcessPoolExecutor(max_workers=number_of_cores) as executor:
        dt_futures = []
        ada_futures = []
        for dt_setting, ada_setting in zip(dt_settings_queue, ada_settings_queue):
            dt_futures.append(executor.submit(
                decision_trees.decision_tree_classification, *dt_setting))
            ada_futures.append(executor.submit(
                boosting_dt.AdaBoost_classification,*ada_setting))

        if len(dt_settings_queue) > len(ada_settings_queue):
            for setting in dt_settings_queue[len(ada_settings_queue):]:
                dt_futures.append(executor.submit(
                    decision_trees.decision_tree_classification, *setting))

        if len(ada_settings_queue) > len(dt_settings_queue):
            for setting in ada_settings_queue[len(dt_settings_queue):]:
                ada_futures.append(executor.submit(
                    boosting_dt.AdaBoost_classification, *setting))
        
        print(f"Waiting for DT and ADA to finish...")
        dt_results = {}
        for future in dt_futures:
            try:
                result = future.result()  
                dt_results[result['setting_code']] = result 
            except Exception as e:
                print("Error in Decision Tree task:", e)

        ada_results = {}
        for future in ada_futures:
            try:
                result = future.result() 
                ada_results[result['setting_code']] = result 
            except Exception as e:
                print("Error in AdaBoost task:", e)
        print(f"DT and ADA finished")

    print(f"Start running Random Forest...")
    # Now that we've got all the cores free again, we can run Random Forest
    rf_results = {}
    for i, rf_setting in enumerate(rf_settings_queue):
        rf_classifier, X_test, y_test, y_pred, doubted_rows, delta_t, feat_imp, setting_code = \
            bagging_dt.random_forest_classification(*rf_setting)
        rf_results[setting_code] = rf_classifier, X_test, y_test, y_pred, doubted_rows, delta_t, feat_imp

    print(f"Random Forest finished\nMaking plots...")
    utils.make_plots(dt_results, rf_results, ada_results)
    print(f"Plots made")
    print(f"Done")
    
        

