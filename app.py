import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Function to convert time string M:SS to total seconds
def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

# Load the trained model
model_path = os.path.join('fight_model.pkl')
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")

# Load and prepare fight data
try:
    fight_data_path = os.path.join('new_fight_detail_full.csv')
    fight_data = pd.read_csv(fight_data_path)

    # Convert 'Time' from M:SS to seconds
    fight_data['Time'] = fight_data['Time'].apply(time_to_seconds)

    # Apply one-hot encoding to categorical columns
    fight_data = pd.get_dummies(fight_data, columns=['Weight Class', 'Winning Method', 'Win/Loss (Fighter1)'])

except Exception as e:
    st.error(f"Failed to load or prepare fight data: {str(e)}")

# Encode fighter names
encoder = LabelEncoder()
try:
    all_fighters = pd.concat([fight_data['Fighter1'], fight_data['Fighter2']]).unique()
    encoder.fit(all_fighters)
    fight_data['Fighter1'] = encoder.transform(fight_data['Fighter1'])
    fight_data['Fighter2'] = encoder.transform(fight_data['Fighter2'])
except Exception as e:
    st.error(f"Failed to encode fighter names: {str(e)}")

# App title
st.title('Fight Win Predictor')

# Selecting a weight class (filter now uses one-hot encoded columns)
try:
    # Extract original weight classes before encoding
    weight_classes = [col.split('_')[-1] for col in fight_data.columns if 'Weight Class_' in col]
    selected_weight_class = st.selectbox('Select Weight Class', weight_classes)

    # Filter data based on selected weight class (using encoded column)
    filtered_fight_data = fight_data[fight_data[f'Weight Class_{selected_weight_class}'] == 1]
except Exception as e:
    st.error(f"Failed to setup weight class selection: {str(e)}")

# Setup fighter dropdowns and display stats
col1, col2 = st.columns(2)
try:
    filtered_fighters1 = filtered_fight_data['Fighter1'].unique()
    filtered_fighters2 = filtered_fight_data['Fighter2'].unique()

    with col1:
        fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters1, format_func=lambda x: encoder.inverse_transform([x])[0])
        fighter1_stats = filtered_fight_data[filtered_fight_data['Fighter1'] == fighter1].iloc[0]
        st.write('Fighter 1 Stats:', fighter1_stats)
    with col2:
        fighter2 = st.selectbox('Select Fighter 2', options=filtered_fighters2, format_func=lambda x: encoder.inverse_transform([x])[0])
        fighter2_stats = filtered_fight_data[filtered_fight_data['Fighter2'] == fighter2].iloc[0]
        st.write('Fighter 2 Stats:', fighter2_stats)
except Exception as e:
    st.error(f"Failed to display fighter options or stats: {str(e)}")

# Predict button
if st.button('Predict Outcome'):
    try:
        # Prepare input data for prediction, ensuring no non-numeric data remains
        input_features = filtered_fight_data.drop(['Fighter1', 'Fighter2'], axis
