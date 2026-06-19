import joblib
import numpy as np

# Load model
model = joblib.load("models/stroke_model.pkl")

# Example input:
# age, hypertension, heart_disease, avg_glucose_level, bmi, ...
sample_data = np.array([[45, 1, 0, 150.5, 28.4, 1, 1, 2, 1, 2]])

prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("High Risk of Stroke")
else:
    print("Low Risk of Stroke")
