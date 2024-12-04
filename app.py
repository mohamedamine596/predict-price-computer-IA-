import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def preprocess_input(data):
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
    
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['Company', 'Type Name', 'Cpu_brand', 'Gpu_brand', 'Os'])
    
    # Add missing columns
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0
    
    # Reorder columns to match the required features
    data = data[required_features]
    
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
    company = st.selectbox('Company', ['Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi'])
    type_name = st.selectbox('Type Name', ['Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation'])
    ram = st.number_input('RAM (GB)', min_value=1, max_value=64, step=1)
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touch_screen = st.selectbox('Touch Screen', ['Yes', 'No'])
    ips = st.selectbox('IPS', ['Yes', 'No'])
    ppi = st.number_input('PPI', min_value=50, max_value=500, step=1)
    cpu_brand = st.selectbox('CPU Brand', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor'])
    hdd = st.number_input('HDD (GB)', min_value=0, max_value=2048, step=1)
    ssd = st.number_input('SSD (GB)', min_value=0, max_value=2048, step=1)
    gpu_brand = st.selectbox('GPU Brand', ['Intel', 'Nvidia'])
    os = st.selectbox('OS', ['Others', 'Windows'])
    
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
        st.success(f'Predicted Price: ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
