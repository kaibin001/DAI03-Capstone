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

# Load datasets
fighters_df = pd.read_csv('new_fight_detail_full.csv')

# Convert percentage columns if necessary
# fighters_df['SomeColumn'] = fighters_df['SomeColumn'].str.rstrip('%').astype('float') / 100

# Streamlit user interface
st.title('UFC Fight Predictor')

# Selecting fighters
fighter1 = st.selectbox('Select Fighter 1', options=fighters_df['Fighter1'].unique())
fighter2 = st.selectbox('Select Fighter 2', options=[f for f in fighters_df['Fighter2'].unique() if f != fighter1])

# Show stats
if st.button('Show Stats'):
    st.write('Fighter 1 Stats:', fighters_df[fighters_df['Fighter1'] == fighter1])
    st.write('Fighter 2 Stats:', fighters_df[fighters_df['Fighter2'] == fighter2])

# Prediction
if st.button('Predict Fight Outcome'):
    # Assume function `prepare_input` processes the input features necessary for the model
    input_features = prepare_input(fighter1, fighter2, fighters_df)
    prediction = model.predict([input_features])[0]
    winner = 'Fighter 1' if prediction == 1 else 'Fighter 2'
    st.success(f'Predicted Winner: {winner}')

def prepare_input(fighter1, fighter2, df):
    # Process and return the input features from dataframe for the model prediction
    # Example placeholder
    return np.array([0])  # Replace with actual feature extraction logic

