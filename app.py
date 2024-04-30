import streamlit as st
import pandas as pd
import pickle

# Load the model and data
model = pickle.load(open('fight_model.pkl', 'rb'))
fighter_details = pd.read_csv('File 3.csv')

# Application Title
st.title("UFC Fight Predictor")

# Fighter Selection
fighter_names = sorted(fighter_details['Full Name'].dropna().unique())
fighter1 = st.selectbox("Select Fighter 1", [''] + fighter_names)
fighter2 = st.selectbox("Select Fighter 2", [''] + [f for f in fighter_names if f != fighter1])

# Display Fighter Stats
if fighter1:
    st.write("Fighter 1 Stats", fighter_details[fighter_details['Full Name'] == fighter1])
if fighter2:
    st.write("Fighter 2 Stats", fighter_details[fighter_details['Full Name'] == fighter2])

# Prediction Button
if st.button("Predict Winner"):
    if not fighter1 or not fighter2:
        st.error("Please select both fighters.")
    else:
        # Assuming you have a function to process input features
        input_features = process_features(fighter1, fighter2, fighter_details)
        prediction = model.predict([input_features])
        winner = "Fighter 1 Wins" if prediction == 1 else "Fighter 2 Wins"
        st.success(winner)

def process_features(fighter1, fighter2, df):
    # You need to replace this part with actual feature processing based on your model
    # This is just a placeholder
    features = []
    return features

