import numpy as np
import pickle
import streamlit as st

#loading_save_model
loaded_model=pickle.load(open("Customer_churn_prediction.sav",'rb'))

#creating_a_function
def Customer_churn_prediction(input_data):
  #changing input_data into array
  input_data_as_array=np.asarray(input_data)
  #reshaping the array
  input_data_reshaped=input_data_as_array.reshape(1,-1)
  #prediction
  prediction=loaded_model.predict(input_data_reshaped)
  print(prediction)
  #condition
  if(prediction[0]==0):
    print("The person will not exit")
  else:
    print("The person will exit")
    

def main():
    
    #title
    st.title("ChurnWhisperers")
    
    #input_data    
    CreditScore=st.text_input("Enter CreditScore")
    Age=st.text_input("Enter Your Age")
    Tenure=st.text_input("Enter your Tenure")
    Balance=st.text_input("Enter your Balance")
    NumOfProducts=st.text_input("Enter Num Of Products you owned")
    HasCrCard=st.text_input("Do you have Credit card")
    IsActiveMember=st.text_input("you are a active member(1,0)")
    EstimatedSalary=st.text_input("Enter Estimated Salary")
    Geography_Germany=st.text_input("you live in Germany(True/False)")
    Geography_Spain=st.text_input("you live in spain(True/False)")
    Gender_Male=st.text_input("You are a male(True/False)")
    
    #code_for_prediction
    predict=""
    
    #create_button for results
    
    if st.button("Results:"):
        predict=Customer_churn_prediction([CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography_Germany,Geography_Spain,Gender_Male])
    
    st.success(predict)
    
    
if __name__=='__main__':
    main() 
    
    
