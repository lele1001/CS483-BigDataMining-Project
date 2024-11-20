import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class DiabetesPredictor:
    def __init__(self, model_path="../models/FNN.h5", csv_path='../data/balanced.csv', sample_size=100, feature_names=[]):
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.feature_names = ["Has High Blood Pressure (0 no, 1 yes)", "Has High Cholesterole (0 no, 1 yes)", "Does Cholesterole Check (0 no, 1 yes)", "Body Mass Index", "Is a Smoker (0 no, 1 yes)", "Had a Stroke (0 no, 1 yes)", "Has an Heart Disease (0 no, 1 yes)", "Does Physical Activity (0 no, 1 yes)", "Eats Fruits (0 no, 1 yes)", "Eats Veggies (0 no, 1 yes)", "High Alchol Consume (0 no, 1 yes)", "Has Health Insurance (0 no, 1 yes)", "Can pay for a Doctor (0 yes, 1 no)", "General Health Coeffient", "Mental Health Coeffient", "Physical Health Coeffient", "Has difficulty in Walking (0 no, 1 yes)", "Sex of the patient (0 female, 1 male)", "Age Coefficent", "Education Level Coeffient", "Income Level Coeffient"]
        self.shap_explainer = None

        # Load baseline sample data for SHAP computation
        self.baseline_data = self.load_sample_data()
        self.compute_shap_explainer()

    def get_feature_names(self):
        return self.feature_names
    
    def preprocessing(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def load_sample_data(self):
        # Load data from CSV and select a random sample
        full_data = pd.read_csv(self.csv_path)
        sample_data = full_data.sample(n=self.sample_size, random_state=42) 
        return self.preprocessing(sample_data.to_numpy())

    def compute_shap_explainer(self):
        # Initialize the SHAP DeepExplainer with the baseline data
        self.shap_explainer = shap.DeepExplainer(self.model, self.baseline_data)

    def predict_and_interpret(self, patient_data):
        patient_data = self.preprocessing(np.array(patient_data[1:]).reshape((1, -1)))

        probability = float(self.model.predict(patient_data)[0][0])
        shap_values_patient = self.shap_explainer.shap_values(patient_data)[0]
        abs_shap_values = np.abs(shap_values_patient)
        
        # Sort features and SHAP values by absolute contribution
        sorted_indices = np.argsort(abs_shap_values)[::-1]
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_contributions = [abs_shap_values[i] for i in sorted_indices]

        good_features = [
            {"feature": sorted_features[i], "contribution": sorted_contributions[i]}
            for i in range(3)
        ]
        bad_features = [
            {"feature": sorted_features[i], "contribution": sorted_contributions[i]}
            for i in range(-3, 0)
        ]

        result = {
            "prediction": probability,
            "good_features": good_features,
            "bad_features": bad_features,
            "average_impact": np.mean(abs_shap_values),
        }

        return result


data_path = '../data/balanced.csv'

data = pd.read_csv(data_path)
random_row = data.sample(n=1, random_state=42)
patient_data = list(random_row.to_numpy().flatten())

predictor = DiabetesPredictor()

result = predictor.predict_and_interpret(patient_data)
print(result)