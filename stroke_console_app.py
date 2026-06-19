import joblib
import numpy as np
import matplotlib.pyplot as plt

# LOAD MODEL
model = joblib.load("models/stroke_model.pkl")

print("\nBrain Stroke Risk Prediction System\n")

# USER INPUT
sex = int(input("Sex (Male=1, Female=0): "))
hypertension = int(input("Hypertension (0/1): "))
heart_disease = int(input("Heart Disease (0/1): "))
ever_married = int(input("Ever Married (Yes=1, No=0): "))
work_type = int(input("Work Type (Private=0, Self=1, Govt=2, Children=3, Never=4): "))
residence = int(input("Residence (Urban=1, Rural=0): "))
avg_glucose = float(input("Average Glucose Level: "))
bmi = float(input("BMI: "))
smoking = int(input("Smoking (Never=0, Formerly=1, Current=2, Unknown=3): "))
physical = float(input("Physical Activity: "))
alcohol = float(input("Alcohol Intake: "))
stress = float(input("Stress Level: "))
bp = float(input("Blood Pressure: "))
chol = float(input("Cholesterol: "))
family = int(input("Family History (Yes=1, No=0): "))
mri = float(input("MRI Result: "))

# PREPARE INPUT
input_data = np.array([[sex, hypertension, heart_disease, ever_married,
                        work_type, residence, avg_glucose, bmi,
                        smoking, physical, alcohol, stress,
                        bp, chol, family, mri]])

# PREDICT
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]
risk_percent = int(probability * 100)

print("\nStroke Risk Percentage:", risk_percent, "%")

if prediction[0] == 1:
    print("HIGH RISK OF STROKE")
else:
    print("LOW RISK OF STROKE")

# GRAPH 1
parameters = ['Glucose', 'BP', 'Cholesterol', 'Stress']
values = [avg_glucose, bp, chol, stress]
normal = [100, 120, 200, 5]

x = np.arange(len(parameters))

plt.figure()
plt.bar(x - 0.2, values, 0.4, label='Patient')
plt.bar(x + 0.2, normal, 0.4, label='Normal')
plt.xticks(x, parameters)
plt.title("Health Parameter Comparison")
plt.legend()
plt.show()

# GRAPH 2
plt.figure()
plt.pie([risk_percent, 100-risk_percent],
        labels=['Risk', 'Safe'],
        autopct='%1.1f%%')
plt.title("Stroke Risk Distribution")
plt.show()

