from helper import DiabetesPredictor, generate_patient_report
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

baseline_sample_size = 250
data_path = 'data/balanced.csv'
data = pd.read_csv(data_path)
numeric_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
scaler = MinMaxScaler()
data.loc[:, numeric_features] = scaler.fit_transform(data[numeric_features])

predictor = DiabetesPredictor(data.sample(n=baseline_sample_size, random_state=42))

print("\n\nWelcome to the Patient Health Analysis System\n")
print("This system generates a professional report for a patient based on the Diabetes Predictor model\n")
print("Startring the system...\n\n\n")

while True:
    random_row = data.sample(n=1)
    row_index = random_row.index[0]
    print(f"Generating report for patient ID: {row_index}\n")
    patient_data = list(random_row.to_numpy().flatten())

    result = predictor.predict_and_interpret(patient_data)
    #print(result)

    report = generate_patient_report(result, patient_data, predictor.get_feature_names(), row_index)
    #print(report)

    # Save the report to a file
    with open("patient_report.txt", "w") as file:
        file.write(report)
    print("\n\nPatient report saved to 'patient_report.txt'.")

    s = input("\nDo you want to generate another report? (y/n): ")
    if s != "y":
        break
    print("\n")

print("\n\nExiting...\n")