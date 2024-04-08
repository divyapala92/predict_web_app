# IMPORT STATEMENTS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv(r"C:\Users\divya\OneDrive\Desktop\ML web app\Clean_BDHS_Diabetic_Data_Jahan.csv")

# Label Encoding
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Splitting the dataset into training and test sets
X = df.drop(columns=['diabetes'])
y = df['diabetes']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying oversampling only to the training set
ros = RandomOverSampler(sampling_strategy="not majority")
x_train_res, y_train_res = ros.fit_resample(x_train, y_train)

# Train the model using the oversampled data
rf = RandomForestClassifier()
rf.fit(x_train_res, y_train_res)

# Get feature importances
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Select the top N important features
top_n = 5  # Keep the top 5 features
top_features = feature_importance_df['Feature'][:top_n].values

# Output the top 5 features used in the model
print("Top 5 features used in the model:")
print(top_features)

# Retain only the top N features in your training and testing data
x_train_top = x_train_res[top_features]
x_test_top = x_test[top_features]

# Retrain your model using only the top N features
rf_top = RandomForestClassifier()
rf_top.fit(x_train_top, y_train_res)

# Predict on the test set using the top features and calculate accuracy
y_pred_top = rf_top.predict(x_test_top)
accuracy_top = accuracy_score(y_test, y_pred_top)

# Streamlit UI
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')

# Display statistics for the top 5 features
st.subheader('Training Data Stats for Top Features')
st.write(df[top_features].describe())

def user_report(top_features):
    user_report_data = {}
    for feature in top_features:
        # Default value for slider is set to the mean of the feature
        user_report_data[feature] = st.sidebar.slider(f'{feature}', 
                                                      float(df[feature].min()), 
                                                      float(df[feature].max()), 
                                                      float(df[feature].mean()))
    return pd.DataFrame([user_report_data])

user_data = user_report(top_features)
st.subheader('Patient Data')
st.write(user_data)

# Use the trained model 'rf_top' for prediction on the top features
user_result = rf_top.predict(user_data)

# Displaying the prediction result
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Displaying the model accuracy using the top features
st.subheader('Model Accuracy on Test Data with Top Features: ')
st.write(f'{accuracy_top * 100:.2f}%')

# VISUALISATIONS
st.title('Visualised Patient Report')

# Color based on prediction
color = 'red' if user_result[0] == 1 else 'blue'

# Age vs height
st.header('Height Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x='age', y='height', data=df, hue='diabetes', palette='rainbow')
ax12 = sns.scatterplot(x=user_data['age'], y=user_data['height'], s=150, color=color)
plt.xticks(np.arange(35, 51, 1))
plt.yticks(np.arange(1.3, 1.9, 0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs weight
st.header('Weight Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x='age', y='weight', data=df, hue='diabetes', palette='rainbow')
ax12 = sns.scatterplot(x=user_data['age'], y=user_data['weight'], s=150, color=color)
plt.xticks(np.arange(35, 51, 1))
plt.yticks(np.arange(35, 101, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs SBP
st.header('Systolic Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='age', y='SBP', data=df, hue='diabetes', palette='Reds')
ax6 = sns.scatterplot(x=user_data['age'], y=user_data['SBP'], s=150, color=color)
plt.xticks(np.arange(35, 51, 1))
plt.yticks(np.arange(70, 211, 20))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs DBP
st.header('Diastolic Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='age', y='DBP', data=df, hue='diabetes', palette='Reds')
ax6 = sns.scatterplot(x=user_data['age'], y=user_data['DBP'], s=150, color=color)
plt.xticks(np.arange(35, 51, 1))
plt.yticks(np.arange(35, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


  






