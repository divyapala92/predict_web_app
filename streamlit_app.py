#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Read the dataset
df = pd.read_csv("Diabetes_Balanced_Predict.csv")

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x=df.drop("diabetes",axis=1)
y=df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    height = st.sidebar.slider('height',  1.324000, 1.872, 1.659)
    weight = st.sidebar.slider('weight', 35.400002, 96.500, 53.000)
    age = st.sidebar.slider('age', 35.0, 49.0, 43.000)
    DBP = st.sidebar.slider('DBP', 36.0, 129.0, 65.0)
    SBP = st.sidebar.slider('SBP', 72.0, 210.0, 111.0)

    
    user_report = {
        'height': height,
        'weight': weight,
        'age': age,
        'DBP': DBP,
        'SBP': SBP,    
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# Predict using the user_data
user_result = rf.predict(user_data)


# OUTPUT
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'

st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')



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
st.header('Dialostic Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='age', y='DBP', data=df, hue='diabetes', palette='Reds')
ax6 = sns.scatterplot(x=user_data['age'], y=user_data['DBP'], s=150, color=color)
plt.xticks(np.arange(35, 51, 1))
plt.yticks(np.arange(35, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)
