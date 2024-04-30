import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join('fight_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load fighter names and stats
file_3_data_path = os.path.join('File 3.csv')
fighter_names = pd.read_csv(file_3_data_path)

# Convert all columns to string to avoid serialization issues
fighter_names = fighter_names.astype(str)

# App title
st.title('Fight Win Predictor')

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Fighter 1', options=['Please select a fighter'] + sorted(fighter_names['Full Name'].tolist()), key='f1')
    if fighter1 != 'Please select a fighter':
        fighter1_stats = fighter_names[fighter_names['Full Name'] == fighter1].iloc[0]
        st.write('Fighter 1 Stats:', fighter1_stats)
with col2:
    # Exclude selected Fighter 1 from Fighter 2's options
    possible_fighters_2 = sorted(fighter_names[fighter_names['Full Name'] != fighter1]['Full Name'].tolist())
    fighter2 = st.selectbox('Select Fighter 2', options=['Please select a fighter'] + possible_fighters_2, key='f2')
    if fighter2 != 'Please select a fighter':
        fighter2_stats = fighter_names[fighter_names['Full Name'] == fighter2].iloc[0]
        st.write('Fighter 2 Stats:', fighter2_stats)

# Predict button
if st.button('Predict Outcome'):
    if fighter1 != 'Please select a fighter' and fighter2 != 'Please select a fighter':
        # Prepare data for prediction
        # Note: You'll need to properly prepare the feature vector here based on how your model was trained
        input_data = np.array([fighter1_stats.values[:-1] + fighter2_stats.values[:-1]])  # Adjust as per actual feature requirements
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    else:
        st.error("Please select both fighters.")

