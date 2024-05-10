# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data from CSV file using pandas
data = pd.read_csv("spambase_csv.csv")

# Split data into features (X) and target variable (y)
X = data.drop(columns=['class'])  # Replace 'target_column' with the name of your target column
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize SVM classifier
classifier = SVC(kernel='rbf', random_state=42) # You can choose different kernels like 'rbf', 'poly', etc.

# Train the SVM model
classifier.fit(X_train, y_train)

# Predict the labels of test data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import numpy as np
import matplotlib.pyplot as plt



# Plot each instance based on target label
spam_data = data[data['word_freq_all']]
non_spam_data = data[data['class'] == 0]

plt.figure(figsize=(10, 6))
plt.scatter(spam_data.index, spam_data['word_freq_all'], color='red', label='Spam')
plt.scatter(non_spam_data.index, non_spam_data['class'], color='blue', label='Non-Spam')
plt.xlabel('Instance Index')
plt.ylabel('Class')
plt.title('Instances based on Target Label')
plt.legend()
plt.show()
