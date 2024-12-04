import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

## Function to preprocess input data
def preprocess_input(data):
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['Company', 'Type Name', 'Cpu_brand', 'Gpu_brand', 'Os'])
    
    # Ensure all required features are present
    required_features = [
        'Ram', 'Weight', 'TouchScreen', 'Ips', 'Ppi', 'HDD', 'SSD',
        'Company_Apple', 'Company_Asus', 'Company_Chuwi', 'Company_Dell', 'Company_Fujitsu',
        'Company_Google', 'Company_HP', 'Company_Huawei', 'Company_LG', 'Company_Lenovo',
        'Company_MSI', 'Company_Mediacom', 'Company_Microsoft', 'Company_Razer', 'Company_Samsung',
        'Company_Toshiba', 'Company_Vero', 'Company_Xiaomi',
        'TypeName_Gaming', 'TypeName_Netbook', 'TypeName_Notebook', 'TypeName_Ultrabook', 'TypeName_Workstation',
        'Cpu_brand_Intel Core i3', 'Cpu_brand_Intel Core i5', 'Cpu_brand_Intel Core i7', 'Cpu_brand_Other Intel Processor',
        'Gpu_brand_Intel', 'Gpu_brand_Nvidia', 'Os_Others', 'Os_Windows'
    ]
    
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0
    
    # Convert any remaining object columns to category type
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        data[col] = data[col].astype('category')
    
    return data

# Function to make prediction
def predict_price(input_data):
    # Preprocess the input data
    processed_input = preprocess_input(input_data)
    
    # Make prediction
    predicted_price = model.predict(processed_input)
    
    return predicted_price

# Streamlit app
def main():
    st.title('Product Price Prediction')
    
    # Input fields
    company = st.text_input('Company')
    type_name = st.text_input('Type Name')
    ram = st.number_input('RAM')
    weight = st.number_input('Weight')
    touch_screen = st.selectbox('Touch Screen', ['Yes', 'No'])
    ips = st.selectbox('IPS', ['Yes', 'No'])
    ppi = st.number_input('PPI')
    cpu_brand = st.text_input('CPU Brand')
    hdd = st.number_input('HDD')
    ssd = st.number_input('SSD')
    gpu_brand = st.text_input('GPU Brand')
    os = st.text_input('OS')
    
    # Button to trigger prediction
    if st.button('Predict Price'):
        input_data = pd.DataFrame({
            'Company': [company],
            'Type Name': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            'TouchScreen': [1 if touch_screen == 'Yes' else 0],
            'Ips': [1 if ips == 'Yes' else 0],
            'Ppi': [ppi],
            'Cpu_brand': [cpu_brand],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu_brand': [gpu_brand],
            'Os': [os]
        })
        
        # Make prediction
        predicted_price = predict_price(input_data)
        
        # Display prediction
        st.success(f'Predicted Price: ${predicted_price[0]}')

if __name__ == '__main__':
    main()

