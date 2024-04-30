import streamlit as st
import joblib
import numpy as np

model = joblib.load('oopPicklefile_2602100120.pkl')

def main():
    st.title('Churn Prediction using XGBoost')
    st.write("By Anastasia Putri Hartanti - 2602100120")

    CreditScore = st.number_input('CreditScore', min_value=0.0, max_value=1000.0)
    Geography = st.selectbox('Geography - France: 0, Germany: 1, Spain: 2', [0,1,2])
    Gender = st.selectbox('Gender - Female: 0, Male: 1', [0,1])
    Age = st.slider('Age', 0, 100)
    Tenure = st.slider('Tenure', 0, 10)
    Balance = st.number_input('Balance', min_value=0, max_value=300000)
    NumOfProducts = st.selectbox('NumOfProducts', [1,2,3,4])
    HasCrCard = st.checkbox('I have a Credit Card')
    IsActiveMember = st.checkbox('I am an Active Member')
    EstimatedSalary = number_input = st.number_input('Estimated Sallary', min_value=0.0, max_value=250000.0)

   
    if st.button('Predict the Churn'):
        features = [CreditScore, Geography, Gender, Age, Tenure, Balance, 
                    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
        result = make_prediction(features)
        st.success(f'The churn prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    
    return prediction[0]

if __name__ == '__main__':
    main()

