# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:17:43 2023

@author: LENOVO
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Loading saved model
loaded_model=pickle.load(open('C:/Users/LENOVO/Downloads/trained_model.sav','rb'))

# Creating predection 
def diabetes_predection(input_data) :
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main() :
     # giving Title
     st.title('Diabettes Prediction')
     # giving input data from user
     
     Pregnancies = st.text_input('Number of Pregnancies')
     Glucose = st.text_input('Glucose Value')
     BloodPressure = st.text_input('BloodPressure Value')
     SkinThickness = st.text_input('SkinThickness Value')
     Insulin = st.text_input('Insulin level')
     BMI = st.text_input('BMI Value')
     DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
     Age = st.text_input('Age of the Person')
     
     # Code for predection
     diagnosis=""
     
     if st.button("Diabetes Test Result"):
         diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
         
     st.success(diagnosis)
     
if __name__=='__main__' :
    main()
         
     

















