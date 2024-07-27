import numpy as np
import pickle
import streamlit as st

#loading_save_model
loaded_model=pickle.load(open("Diabetes_trained_model.sav",'rb'))

#creating_a_function
def Diabetes_prediction(input_data):
    #changing input_data into array
    input_data_as_array=np.asarray(input_data)
    #reshaping the array
    input_data_reshaped=input_data_as_array.reshape(1,-1)
    #prediction
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    #condition
    if(prediction[0]==0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic"
  

def main():
    
    #title
    st.title("Diabetes_Prediction")
    
    #input_data    
    Pregnancies=st.text_input("Enter No. of pregnancies")
    Glucose=st.text_input("Enter Glucose level")
    BloodPressure=st.text_input("Enter your Blood pressure(BP)")
    BMI=st.text_input("Enter your BMI(Body-mass-index)")
    Age=st.text_input("Enter your Age")
    
    #code_for_prediction
    Diagnosis=""
    
    #create_button for results
    
    if st.button("Results:"):
        Diagnosis=Diabetes_prediction([Pregnancies,Glucose,BloodPressure,BMI,Age])
    
    st.success(Diagnosis)
    
    
if __name__=='__main__':
    main() 
    
    
