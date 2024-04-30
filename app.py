import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_data(allow_output_mutation=True)
def load_model():
    with open('fight_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Load datasets
fighters_df = pd.read_csv('new_fight_detail_full.csv')

# App title
st.title('UFC Fight Predictor')

# Selecting fighters
fighter1 = st.selectbox('Select Fighter 1', options=['Select a fighter'] + sorted(fighters_df['Fighter1'].unique()))
fighter2 = st.selectbox('Select Fighter 2', options=['Select a fighter'] + sorted(fighters_df['Fighter2'].unique()))

# Filter out the same fighter
if fighter1 in fighters_df['Fighter1'].values:
    fighters_df = fighters_df[fighters_df['Fighter1'] != fighter1]

# Display fighter stats
if fighter1 != 'Select a fighter':
    st.write('Fighter 1 Stats:', fighters_df[fighters_df['Fighter1'] == fighter1])
if fighter2 != 'Select a fighter':
    st.write('Fighter 2 Stats:', fighters_df[fighters_df['Fighter2'] == fighter2])

# Button to make prediction
if st.button('Predict Outcome'):
    if fighter1 != 'Select a fighter' and fighter2 != 'Select a fighter':
        input_features = prepare_input(fighter1, fighter2, fighters_df)
        prediction = model.predict([input_features])[0]
        winner = 'Fighter 1' if prediction == 1 else 'Fighter 2'
        st.success(f'Predicted Winner: {winner}')

def prepare_input(fighter1, fighter2, df):
    # This function should prepare the features from your dataset for the model
    # Extract relevant features and perform any required preprocessing
    # This is a placeholder; you'll need to adapt it based on your model's needs
    features = np.array([0])  # Replace with actual feature extraction logic
    return features
