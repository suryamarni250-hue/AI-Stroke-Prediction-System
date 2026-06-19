import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/stroke.csv")

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# -------------------------------
# 1️⃣ Stroke vs Non-Stroke Graph
# -------------------------------
plt.figure()
df['Stroke'].value_counts().plot(kind='bar')
plt.title("Stroke vs Non-Stroke Distribution")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.savefig("stroke_distribution.png")
plt.show()

# -------------------------------
# 2️⃣ Age Distribution Histogram
# -------------------------------
plt.figure()
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")
plt.show()

# -------------------------------
# 3️⃣ Glucose Distribution
# -------------------------------
plt.figure()
sns.histplot(df['Average_Glucose'], kde=True)
plt.title("Glucose Level Distribution")
plt.xlabel("Average Glucose")
plt.ylabel("Frequency")
plt.savefig("glucose_distribution.png")
plt.show()

# -------------------------------
# 4️⃣ Correlation Heatmap
# -------------------------------
plt.figure()
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

print("Graphs generated and saved successfully!")