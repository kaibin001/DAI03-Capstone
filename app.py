import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('fight_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load fighter names and stats
fighter_names = pd.read_csv('File 3.csv')
fighter_stats = pd.read_csv('new_fight_detail_full.csv')

# App title
st.title('UFC Fight Predictor')

# Dropdown for selecting fighters
fighter1 = st.selectbox('Select Fighter 1', ['Select a fighter'] + sorted(fighter_names['Full Name'].unique()))
fighter2 = st.selectbox('Select Fighter 2', ['Select a fighter'] + sorted(fighter_names['Full Name'].unique()))

# Filter out the same fighter
if fighter1 in fighter_names['Full Name'].values:
    fighter_names = fighter_names[fighter_names['Full Name'] != fighter1]

# Display fighter stats
if fighter1 != 'Select a fighter':
    st.write(fighter_stats[fighter_stats['Full Name'] == fighter1])
if fighter2 != 'Select a fighter':
    st.write(fighter_stats[fighter_stats['Full Name'] == fighter2])

# Button to make prediction
if st.button('Predict Outcome'):
    if fighter1 != 'Select a fighter' and fighter2 != 'Select a fighter':
        # Example: assuming your model takes some specific inputs
        input_features = prepare_features(fighter1, fighter2, fighter_stats)
        prediction = model.predict([input_features])
        winner = 'Fighter 1' if prediction == 1 else 'Fighter 2'
        st.success(f'Predicted Winner: {winner}')
    else:
        st.error('Please select both fighters.')

def prepare_features(fighter1, fighter2, stats):
    # This function should prepare the features from your dataset for the model
    # Example implementation needed based on how your model was trained
    return np.array([])
