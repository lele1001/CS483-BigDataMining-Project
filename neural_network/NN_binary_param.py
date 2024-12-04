import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

file_path = '../data/not_balanced.csv'

data = pd.DataFrame()
try:
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}...")
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                   'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
                   'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                   'NoDocbcCost', 'DiffWalk', 'Sex']
numeric_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

scaler = MinMaxScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    '0': Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # sigmoid
    ]),
    # Now a bigger one
    '1': Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        
        Dense(1, activation='sigmoid')
    ]),
    '2': Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ]),
    # Now a smaller one, which we'll compare to the best performing one
    '3': Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ]),
    # Extensible...
}

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

def metrics_to_str(metrics):
    return f"Recall: {metrics[0]}, F1 Score: {metrics[1]},"+\
         f" Precision: {metrics[2]}, Accuracy: {metrics[3]}"

best_metrics = (0.0, 0.0, 0.0, 0.0)
best_model_index = -1
best_models_class_report = None
best_model = None
os.makedirs('../models', exist_ok=True)
best_model_path = '../models/FNN.h5'
base_performance = None
for model_index, model in models.items():
    print(f"Training model {model_index}...")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy for model {model_index}: {test_accuracy}")
    
    # Save the model
    #tf.keras.models.save_model(model, f'../results/NN_binary_{model_index}.h5')
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
    disp.plot()
    '''(tn, fp), (fn, tp) = cm
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)'''

    # Now we want to keep track of the best model, optimizing the trade-off between these metrics
    # Since it is a classification that deals with health, we want to minimize false negatives,
    # so we will use the recall as the main metric
    '''metrics = (accuracy, recall, precision, f1_score)
    best_metrics_old = best_metrics
    best_metrics = get_best(best_metrics, metrics)

    if best_metrics != best_metrics_old:
        best_model = model
        best_model_index = model_index
        best_models_class_report = classification_report(y_test, y_pred)
    '''
    # doing that with weighted averages...
    class_report = classification_report(y_test, y_pred, output_dict=True)
    metrics = get_metrics(class_report)
    # weighted_avg = class_report['weighted avg']
    # accuracy = class_report['accuracy']
    # metrics = (weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1-score'], accuracy) 

    if model_index == '3':
        base_performance = class_report
    
    best_metrics_old = best_metrics
    best_metrics = get_best(best_metrics, metrics)

    if best_metrics != best_metrics_old:
        best_model = model
        best_model_index = model_index
        best_models_class_report = classification_report(y_test, y_pred)
    
    os.makedirs('confusion_matrices', exist_ok=True)
    plt.savefig(f'confusion_matrices/NN_binary_{model_index}_confusion_matrix.png')
    #plt.show()
    
    # Classification report
    print(classification_report(y_test, y_pred))

print(f"Best model (index): {best_model_index}")
print(f"\tBest model's description: {models[best_model_index].summary()}")
print(f"Best metrics: {metrics_to_str(best_metrics)}")
print(f"Saving best model to {best_model_path}...")
tf.keras.models.save_model(best_model, best_model_path)
print(f"Model saved successfully to {best_model_path}...")
print(f"Classification report for best model: {best_models_class_report}")
print(f"Saving best model classification report to {best_model_path.replace('.h5', '.txt')}...")
with open(best_model_path.replace('.h5', '.txt'), 'w') as f:
    f.write(best_models_class_report)
print(f"Classification report saved successfully to {best_model_path.replace('.h5', '.txt')}...")
print(f"As a reference, the base model (index 3) had the following performance:")
print(f"Classification report for base model: {base_performance}")
print("Done!")