import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("Customer_churn_prediction.sav", 'rb'))

# Creating a function for prediction
def Customer_churn_prediction(input_data):
    # Changing input_data into array
    input_data_as_array = np.asarray(input_data, dtype=np.float64)
    # Reshaping the array
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    # Prediction
    prediction = loaded_model.predict(input_data_reshaped)
    # Condition
    if prediction[0] == 0:
        return "The person will not exit"
    else:
        return "The person will exit"

def main():
    # Title
    st.title("ChurnWhisperers")
    
    # Input data
    CreditScore = st.text_input("Enter CreditScore")
    Age = st.text_input("Enter Your Age")
    Tenure = st.text_input("Enter your Tenure")
    Balance = st.text_input("Enter your Balance")
    NumOfProducts = st.text_input("Enter Num Of Products you own")
    HasCrCard = st.text_input("Do you have a Credit card (0 for No, 1 for Yes)")
    IsActiveMember = st.text_input("Are you an active member (0 for No, 1 for Yes)")
    EstimatedSalary = st.text_input("Enter Estimated Salary")
    Geography_Germany = st.text_input("Do you live in Germany (0 for No, 1 for Yes)")
    Geography_Spain = st.text_input("Do you live in Spain (0 for No, 1 for Yes)")
    Gender_Male = st.text_input("Are you male (0 for No, 1 for Yes)")
    
    # Code for prediction
    predict = ""
    
    # Create button for results
    if st.button("Results"):
        input_data = [
            CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, 
            IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male
        ]
        
        # Convert inputs to appropriate data types
        input_data = list(map(float, input_data))
        
        # Get prediction
        predict = Customer_churn_prediction(input_data)
    
    st.success(predict)

if __name__ == '__main__':
    main()
