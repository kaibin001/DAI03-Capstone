import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load fighter data and model
fighters_df = pd.read_csv('File 3.csv')
model_path = 'fight_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# App title
st.title("UFC Fight Prediction")

# Dropdown for selecting fighters
fighter1 = st.selectbox("Fighter 1", ["Please select or type in the Fighter's name"] + sorted(fighters_df['Full Name'].unique()))
fighter2 = st.selectbox("Fighter 2", ["Please select or type in the Fighter's name"] + sorted(fighters_df['Full Name'].unique()))

if fighter1 in fighters_df['Full Name'].values:
    fighters_df = fighters_df[fighters_df['Full Name'] != fighter1]
if fighter2 in fighters_df['Full Name'].values:
    fighters_df = fighters_df[fighters_df['Full Name'] != fighter2]

# Display fighter stats
if fighter1 != "Please select or type in the Fighter's name":
    fighter1_stats = fighters_df[fighters_df['Full Name'] == fighter1]
    st.write("Fighter 1 Stats:", fighter1_stats.transpose())

if fighter2 != "Please select or type in the Fighter's name":
    fighter2_stats = fighters_df[fighters_df['Full Name'] == fighter2]
    st.write("Fighter 2 Stats:", fighter2_stats.transpose())

# Predict and display fight outcome
if st.button("Predict Outcome"):
    if fighter1 != "Please select or type in the Fighter's name" and fighter2 != "Please select or type in the Fighter's name":
        # Placeholder for actual prediction logic
        # You would need to encode the selected fighters' details and any required fight parameters
        # For demonstration, this is simplified and may need actual feature engineering
        st.write("Prediction: Fighter 1 Wins!")  # Placeholder prediction
    else:
        st.error("Please select both fighters.")

