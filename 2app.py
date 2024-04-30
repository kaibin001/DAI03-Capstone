import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_path = 'fight_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load fighter data
fighter_data_path = 'File 3.csv'
fighter_data = pd.read_csv(fighter_data_path)

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

def prepare_input_data(fighter1, fighter2, fighter_data):
    # This function should prepare the input data for the model
    # Convert fighter stats to the model's expected input format
    return input_data
