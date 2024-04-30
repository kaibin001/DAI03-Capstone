import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model_path = 'fight_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load fighter data
fighter_data_path = 'File 3.csv'
fighter_data = pd.read_csv(fighter_data_path)

def prepare_input_data(fighter1, fighter2, fighter_data):
    # This function needs to prepare the data format exactly as the model expects
    # Let's assume your model needs numerical stats of the fighters as input
    
    # Example: Extract numeric stats for both fighters
    f1_stats = fighter_data[fighter_data['Full Name'] == fighter1].select_dtypes(include=[np.number])
    f2_stats = fighter_data[fighter_data['Full Name'] == fighter2].select_dtypes(include=[np.number])
    
    # Flatten the arrays and concatenate them
    if not f1_stats.empty and not f2_stats.empty:
        input_data = np.hstack((f1_stats.values.flatten(), f2_stats.values.flatten()))
        input_data = input_data.reshape(1, -1)  # Reshape for a single sample prediction
        return input_data
    else:
        return None
        
# Dropdown to select fighters
fighter_list = sorted(fighter_data['Full Name'].unique())
fighter1 = st.selectbox('Select Fighter 1', ['Select a fighter'] + fighter_list)
fighter2 = st.selectbox('Select Fighter 2', ['Select a fighter'] + fighter_list)

# Filter the second dropdown
if fighter1 in fighter_list:
    fighter_list.remove(fighter1)

# Display fighter stats
if fighter1 != 'Select a fighter':
    st.write(fighter_data[fighter_data['Full Name'] == fighter1])

if fighter2 != 'Select a fighter' and fighter2 != fighter1:
    st.write(fighter_data[fighter_data['Full Name'] == fighter2])

# Button to predict the outcome
if st.button('Predict Fight Outcome'):
    # Logic to prepare input data and make predictions
    # Assume you prepare input_data here
    input_data = prepare_input_data(fighter1, fighter2, fighter_data)  # You need to implement this function
    prediction = model.predict(input_data)
    result = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
    st.success(result)
    

