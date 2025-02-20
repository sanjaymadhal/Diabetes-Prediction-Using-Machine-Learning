import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

#set the page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

#load the saved diabetes model
diabetes_model_path = r"diabetes_model.sav"
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

#header
st.header("Diabetes Prediction Using Machine Learning")

#getting the input data from the user
col1,col2,col3 = st.columns(3)
with col1:
    pregnancies = st.text_input("Number of Pregnancies")
with col2:
    glucose = st.text_input("Glucose Level")
with col3:
    blood_pressure = st.text_input("Blood Pressure Level")
with col1:
    skin_thickness = st.text_input("Skin Thickness")    
with col2:
    insulin = st.text_input("Insulin Level")
with col3:
    bmi = st.text_input("BMI")
with col1:
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function")
with col2:
    age = st.text_input("Age")
with col3:
    submit = st.button("Predict")
    if submit:
        #predict the diabetes
        prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        if prediction[0] == 1:
            st.error("You have Diabetes")
        else:
            st.success("You don't have Diabetes")
with col1:
    if st.button('Check Accuracy'):
        # Load the test data
        test_data = pd.read_csv(r"diabetes.csv")
        
        # Split the data into features (X) and target (y)
        x_test = test_data.drop(columns=['Outcome'])
        y_test = test_data['Outcome']
        
        # Make predictions
        y_pred = diabetes_model.predict(x_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display the accuracy
        st.write(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
                                  

#footer
st.markdown("Created by [Sanjay M D](https://www.linkedin.com/in/sanjay-m-d/)")
