import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data_cleaned = pd.read_csv("Exasens_cleaned.csv")

# Encode the target variable (Diagnosis)
label_encoder = LabelEncoder()
data_cleaned['Diagnosis'] = label_encoder.fit_transform(data_cleaned['Diagnosis'])

# Separate features and target
X = data_cleaned.drop(columns=['Diagnosis'])
y = data_cleaned['Diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target variable to categorical (one-hot encoding)
y_categorical = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build a neural network model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')  # 4 output neurons for the 4 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=8)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy}")

# Define the label mapping manually
label_mapping = {
    0: "Asthma",
    1: "COPD",
    2: "HC",
    3: "Infected"
}

# Streamlit app
st.title("Respiratory Disease Prediction")

# Input fields for user input
imagery_part_min = st.number_input("Imaginary Part Min", value=-300.5)
imagery_part_avg = st.number_input("Imaginary Part Avg", value=-305.45)
real_part_min = st.number_input("Real Part Min", value=-460.35)
real_part_avg = st.number_input("Real Part Avg", value=-470.12)
gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
smoking = st.selectbox("Smoking Status", [1, 2, 3], format_func=lambda x: {1: "Non-smoker", 2: "Ex-smoker", 3: "Active-smoker"}[x])

# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the input data
    new_input = np.array([[imagery_part_min, imagery_part_avg, real_part_min, real_part_avg, gender, age, smoking]])
    
    # Scale the input
    new_input_scaled = scaler.transform(new_input)
    
    # Make a prediction
    prediction = model.predict(new_input_scaled)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_mapping[predicted_class_index]
    
    # Show the result
    st.write(f"Predicted Diagnosis: **{predicted_class}**")
