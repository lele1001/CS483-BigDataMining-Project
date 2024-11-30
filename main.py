from SHAP import DiabetesPredictor, generate_patient_report
import pandas as pd

data_path = 'data/balanced.csv'
data = pd.read_csv(data_path)
predictor = DiabetesPredictor()

print("\n\nWelcome to the Patient Health Analysis System!\n")

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