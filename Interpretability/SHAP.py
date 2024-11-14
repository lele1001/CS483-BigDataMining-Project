import numpy as np
import pandas as pd
import shap
import tensorflow as tf

class DiabetesPredictor:
    def __init__(self, model_path, csv_path, sample_size=100, feature_names=[]):
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.feature_names = feature_names
        self.shap_explainer = None

        # Load baseline sample data for SHAP computation
        self.baseline_data = self.load_sample_data()
        self.compute_shap_explainer()

    def load_sample_data(self):
        # Load data from CSV and select a random sample
        full_data = pd.read_csv(self.csv_path)
        sample_data = full_data.sample(n=self.sample_size, random_state=42) 
        return sample_data.values

    def compute_shap_explainer(self):
        # Initialize the SHAP DeepExplainer with the baseline data
        self.shap_explainer = shap.DeepExplainer(self.model, self.baseline_data)

    def predict_and_interpret(self, patient_data):
        probability = float(self.model.predict(np.array([patient_data]))[0][0])

        shap_values_patient = self.shap_explainer.shap_values(np.array([patient_data]))[0]

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

feature_names = list(data.columns)
random_row = data.sample(n=1, random_state=42)
patient_data = random_row.values.flatten().tolist()

predictor = DiabetesPredictor(
    model_path='model.h5',
    csv_path=data_path,
    sample_size=100,
    feature_names=feature_names
)

result = predictor.predict_and_interpret(patient_data)
print(result)