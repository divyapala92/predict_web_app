# IMPORT STATEMENTS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")

# Label Encoding
le = LabelEncoder()
df['diabetes'] = le.fit_transform(df['diabetes'])

# Define features and target variable based on specified important features
X = df[['weight', 'height', 'SBP', 'DBP', 'age']]
y = df['diabetes']

# Splitting the dataset into training and test sets using train_test_split function from scikit learn
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Initialise and train a random forest classifier using training data
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Use trained model to make predicionst on the test set
y_pred = rf.predict(x_test)

# Calculate the confusion matrix and performance metrics
cm = confusion_matrix(y_test, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Streamlit UI
st.title('Diabetes Checkup')

st.write("## Model Performance Metrics")
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Displaying the confusion matrix and performance metrics
st.write("## Confusion Matrix and Performance Metrics")
st.write("Confusion Matrix:")
st.write(pd.DataFrame({
    "": ["Actual Negative", "Actual Positive"],
    "Predicted Negative": [TN, FN],
    "Predicted Positive": [FP, TP]
}))

st.sidebar.header('Patient Data Input')

# Creating sliders for input features
user_data = {}
for feature in ['weight', 'height', 'SBP', 'DBP']:
    user_data[feature] = st.sidebar.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

# Modify the age slider to cover a wider age range
user_data['age'] = st.sidebar.slider("Age", 35, 49, 40)  # Adjust the age range as needed

user_data_df = pd.DataFrame([user_data])

# Displaying user input features
st.subheader('Patient Data')
st.write(user_data_df)

# Use the trained model for prediction on user input
user_result = rf.predict(user_data_df)

# Displaying the prediction result
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Define color based on prediction result
color = 'red' if user_result[0] == 1 else 'blue'

# VISUALIZATIONS
st.write("## Data Visualizations")

# Determine x-axis limits for visualization based on the minimum and maximum age in the dataset
x_min = min(df['age'])
x_max = max(df['age'])

# Age vs weight
st.header('Weight Value Graph (Others vs Yours)')
fig_weight = plt.figure()
sns.scatterplot(x='age', y='weight', data=df, hue='diabetes', palette='rainbow')
sns.scatterplot(x=[user_data['age']], y=[user_data['weight']], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
plt.xlim(x_min, x_max)  # Adjust x-axis limits dynamically based on dataset
st.pyplot(fig_weight)

# Age vs height
st.header('Height Value Graph (Others vs Yours)')
fig_height = plt.figure()
sns.scatterplot(x='age', y='height', data=df, hue='diabetes', palette='rainbow')
sns.scatterplot(x=[user_data['age']], y=[user_data['height']], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
plt.xlim(x_min, x_max)  # Adjust x-axis limits dynamically based on dataset
st.pyplot(fig_height)

# Age vs SBP
st.header('Systolic Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
sns.scatterplot(x='age', y='SBP', data=df, hue='diabetes', palette='Reds')
sns.scatterplot(x=[user_data['age']], y=[user_data['SBP']], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
plt.xlim(x_min, x_max)  # Adjust x-axis limits dynamically based on dataset
st.pyplot(fig_bp)

# Age vs DBP
st.header('Diastolic Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
sns.scatterplot(x='age', y='DBP', data=df, hue='diabetes', palette='Reds')
sns.scatterplot(x=[user_data['age']], y=[user_data['DBP']], s=150, color=color)
plt.title('0 - Healthy & 1 - Unhealthy')
plt.xlim(x_min, x_max)  # Adjust x-axis limits dynamically based on dataset
st.pyplot(fig_bp)
