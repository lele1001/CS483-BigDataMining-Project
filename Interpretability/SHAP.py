import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class DiabetesPredictor:
    def __init__(self, model_path="../models/FNN.h5", csv_path='../data/balanced.csv', sample_size=100, feature_names=[]):
        # Load the Keras model
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            pass
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.feature_names = ["Has High Blood Pressure (0 no, 1 yes)", "Has High Cholesterole (0 no, 1 yes)", "Does Cholesterole Check (0 no, 1 yes)", "Body Mass Index", "Is a Smoker (0 no, 1 yes)", "Had a Stroke (0 no, 1 yes)", "Has an Heart Disease (0 no, 1 yes)", "Does Physical Activity (0 no, 1 yes)", "Eats Fruits (0 no, 1 yes)", "Eats Veggies (0 no, 1 yes)", "High Alchol Consume (0 no, 1 yes)", "Has Health Insurance (0 no, 1 yes)", "Can pay for a Doctor (0 yes, 1 no)", "General Health Coeffient", "Mental Health Coeffient", "Physical Health Coeffient", "Has difficulty in Walking (0 no, 1 yes)", "Sex of the patient (0 female, 1 male)", "Age Coefficent", "Education Level Coeffient", "Income Level Coeffient"]
        self.shap_explainer = None

        # Load baseline sample data for SHAP computation
        try:
            self.baseline_data = self.load_sample_data()
            self.compute_shap_explainer()
        except  Exception:
            pass

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
        sorted_indices = np.argsort(shap_values_patient)
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_contributions = [abs_shap_values[i] for i in sorted_indices]

        good_features = [
            {"feature": sorted_features[i], "contribution": sorted_contributions[i]}
            for i in range(3)
        ]
        bad_features = [
            {"feature": sorted_features[i], "contribution": sorted_contributions[i]}
            for i in range(-1, -4, -1)
        ]

        result = {
            "prediction": probability,
            "good_features": good_features,
            "bad_features": bad_features,
            "average_impact": np.mean(abs_shap_values),
        }

        return result


# Generates a prompt for the GPT model based on the prediction and feature analysis
def generate_prompt(result, patient_data, feature_names):
    feature_details = "\n".join(
        [f"{name}: {value}" for name, value in zip(feature_names, patient_data)]
    )

    # Extract the top positive and negative features
    good_features = ", ".join(
        [f"{item['feature']} ({item['contribution']:.2f})" for item in result["good_features"]]
    )
    bad_features = ", ".join(
        [f"{item['feature']} ({item['contribution']:.2f})" for item in result["bad_features"]]
    )

    # Generate the prompt
    prompt = f"""
        Patient Analysis Report:
        - Predicted Probability of being Healthy: {result['prediction']:.2f}
        - Average Contribution of Features: {result['average_impact']:.2f}

        Key Positive Features (supporting health): {good_features}
        Key Negative Features (indicating risks): {bad_features}

        Detailed Feature Values:
        {feature_details}

        Generate a professional report explaining the patient's health condition based on the predicted probability and the feature analysis above. Highlight potential areas of improvement and suggestions for a healthier lifestyle.
    """
    return prompt

# Generates a patient report using the GPT model
def generate_patient_report(result, patient_data, feature_names):
    prompt = generate_prompt(result, patient_data, feature_names)

    # Call the model to generate the report
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Stampa o ritorna il report
    print(response.choices[0].message['content'])
    return response.choices[0].message['content']


'''data_path = '../data/balanced.csv'

data = pd.read_csv(data_path)
random_row = data.sample(n=1, random_state=42)
patient_data = list(random_row.to_numpy().flatten())

predictor = DiabetesPredictor()

result = predictor.predict_and_interpret(patient_data)
print(result)'''