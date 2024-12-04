import numpy as np
import shap
import tensorflow as tf
import google.generativeai as genai

API_KEY = "your_api_key"
genai.configure(api_key=API_KEY)

class DiabetesPredictor:
    def __init__(self, baseline_data, model_path="models/NN_5050.h5", weights_path="models/NN_5050.h5", feature_names=[]):
        # Load the Keras model
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model.load_weights(weights_path)
        except Exception:
            raise ValueError(f"Failed to load model from path: {model_path}")
        self.feature_names = [
            "Has High Blood Pressure (0 no, 1 yes)", 
            "Has High Cholesterole (0 no, 1 yes)", 
            "Does Cholesterole Check (0 no, 1 yes)", 
            "Body Mass Index (scale from 0 to 1)", 
            "Is a Smoker (0 no, 1 yes)", 
            "Had a Stroke (0 no, 1 yes)", 
            "Has an Heart Disease (0 no, 1 yes)", 
            "Does Physical Activity (0 no, 1 yes)", 
            "Eats Fruits (0 no, 1 yes)", 
            "Eats Veggies (0 no, 1 yes)", 
            "High Alchol Consume (0 no, 1 yes)", 
            "Has Health Insurance (0 no, 1 yes)", 
            "Can pay for a Doctor (0 yes, 1 no)", 
            "General Health Fragility Coeffient (represent a scale from 0 to 1, the lower the better)", 
            "Mental Health Fragility Coeffient (represent how many days was bad in last month, scaled between 0 and 1)", 
            "Physical Health Fragility Coeffient (represent how many days was bad in last month, scaled between 0 and 1)", 
            "Has difficulty in Walking (0 no, 1 yes)", 
            "Sex of the patient (0 female, 1 male)", 
            "Age Coefficent (scale from 0 to 1)", 
            "Education Level Coeffient (scale from 0 to 1)", 
            "Income Level Coeffient (scale from 0 to 1)"
        ]
        self.shap_explainer = None

        # Load baseline sample data for SHAP computation
        try:
            self.baseline_data = baseline_data.iloc[:, 1:]
            print("\n\n\nBaseline data loaded successfully")
            self.compute_shap_explainer()
        except Exception:
            raise ValueError("Failed to load baseline data for SHAP computation")

    def get_feature_names(self):
        return self.feature_names

    def compute_shap_explainer(self):
        try:
            if self.model is None:
                raise ValueError("Model is not loaded properly.")
            if self.baseline_data is None:
                raise ValueError("Baseline data is not initialized.")
            print("\nInitializing SHAP explainer...\n")
            # Initialize SHAP DeepExplainer
            self.shap_explainer = shap.KernelExplainer(lambda x: self.model.predict(x), self.baseline_data)
            print("\nSHAP explainer successfully initialized")
        except Exception as e:
            print(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None

    def predict_and_interpret(self, patient_data):
        patient_data = np.array(patient_data[1:], dtype=np.float32).reshape((1, -1))
        probability = 1 - float(self.model.predict(patient_data)[0][0])
        print(f"Predicted probability of being healty: {probability}")

        shap_values_patient = np.array(self.shap_explainer(patient_data).values).flatten()
        abs_shap_values = np.abs(shap_values_patient)

        # Ensure feature names align with SHAP values
        if len(self.feature_names) != abs_shap_values.shape[0]:
            raise ValueError(
                f"Mismatch between feature names ({len(self.feature_names)}) and SHAP values ({abs_shap_values.shape[0]})."
            )
        
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
            for i in range(-3, 0)
        ]

        result = {
            "prediction": probability,
            "good_features": good_features,
            "bad_features": bad_features,
            "average_impact": np.mean(abs_shap_values),
        }

        return result


# Generates a prompt for the LLM based on the prediction and feature analysis
def generate_prompt(result, patient_data, feature_names, row_index):
    feature_details = "\n".join(
        [f"{name}: {np.round(value, 2)}" if ("yes" not in name and "Sex" not in name) else f"{name}: {int(value)}" for name, value in zip(feature_names, patient_data)]
    )

    print("\nFeature Details:\n")
    print(feature_details)

    # Extract the top positive and negative features
    good_features = ", ".join(
        [f"{item['feature']} has a confidence value of ({item['contribution']:.2f})" for item in result["good_features"]]
    )
    bad_features = ", ".join(
        [f"{item['feature']} has a confidence value of ({item['contribution']:.2f})" for item in result["bad_features"]]
    )

    # Generate the prompt
    prompt = f"""
        Patient Analysis Report from Diabetes Predictor (Patient ID: {row_index}):
        - Predicted Probability of being Healthy (No Diabete): {result['prediction']:.2f}
        - Average Confidence value of Features for prediction (use to asses the strength of information in Key positive and negative features, but do not mention explicitly): {result['average_impact']:.2f}

        Key Positive Features (supporting health): {good_features}
        Key Negative Features (indicating risks): {bad_features}

        Detailed Feature Values (all between 0 and 1):
        {feature_details}

        Generate a professional report explaining the patient's health condition based on the predicted probability and the feature analysis above. Highlight potential areas of improvement and suggestions for a healthier lifestyle.
        (do not include date, just text no markdown or latex. rank the recommendations based on the feature confidence, but do not include the confidence values in the report)
    """
    return prompt

# Generates a patient report using the GPT model
def generate_patient_report(result, patient_data, feature_names, row_index):
    patient_data = patient_data[1:]  # Remove the first column (label)
    prompt = generate_prompt(result, patient_data, feature_names, row_index)

    # Call the model to generate the report
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Failed to generate report: {e}")
        return "Error in generating the report."


'''data_path = '../data/balanced.csv'

data = pd.read_csv(data_path)
random_row = data.sample(n=1, random_state=42)
patient_data = list(random_row.to_numpy().flatten())

predictor = DiabetesPredictor()

result = predictor.predict_and_interpret(patient_data)
print(result)

report = generate_patient_report(result, patient_data, predictor.get_feature_names())
print(report)

# Save the report to a file
with open("patient_report.txt", "w") as file:
    file.write(report)
print("Patient report saved to 'patient_report.txt'.")'''