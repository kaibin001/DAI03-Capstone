import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join('fight_model.pkl')
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")

# Load fight data and fighter names
fight_data_path = os.path.join('new_fight_detail_full.csv')
fight_data = pd.read_csv(fight_data_path)
file_3_data_path = os.path.join('File 3.csv')
fighter_names = pd.read_csv(file_3_data_path)

# Convert all columns to string to avoid serialization issues
fighter_names = fighter_names.astype(str)
fight_data = fight_data.astype(str)

# App title
st.title('Fight Win Predictor')

# Weight class selection
weight_classes = fight_data['Weight Class'].unique()
selected_weight_class = st.selectbox('Select Weight Class', weight_classes)

# Filter fighters based on selected weight class
filtered_fighters = fight_data[fight_data['Weight Class'] == selected_weight_class]['Fighter1'].unique()

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters, key='f1')
    fighter1_stats = fighter_names[fighter_names['Column_Name_for_Fighter'] == fighter1].iloc[0]
    st.write('Fighter 1 Stats:', fighter1_stats)
with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=[f for f in filtered_fighters if f != fighter1], key='f2')
    fighter2_stats = fighter_names[fighter_names['Column_Name_for_Fighter'] == fighter2].iloc[0]
    st.write('Fighter 2 Stats:', fighter2_stats)

# Predict button
if st.button('Predict Outcome'):
    # Prepare data for prediction (adjust this according to your model's trained features)
    input_data = np.array([fighter1_stats.values[:-1] + fighter2_stats.values[:-1]])  # Adjust as per actual feature requirements
    prediction = model.predict(input_data)
    win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
    st.success(f'Prediction: {win_status}')
