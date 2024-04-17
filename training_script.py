import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

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

# Initialise and train a random forest classifier using training data
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Use trained model to make predictions on the test set
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

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the performance metrics
print("Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Return the performance metrics directly
performance_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
performance_metrics
