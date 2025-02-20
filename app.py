import pickle
import streamlit as st

#set the page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🧑‍⚕")

#load the saved diabetes model
diabetes_model_path = r'C:\Users\sanjay\OneDrive\Desktop\ROOT\Logistic Regression\diabetes_model.sav'
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

#page title
st.title("Diabetes Prediction Using ML")

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
    accuracy = st.button("Check Accuracy")
    if accuracy:
        st.write("Accuracy of the model is 78.57%")                                        

#footer
st.markdown("Created by [Sanjay M D](https://www.linkedin.com/in/sanjay-m-d/)")
