import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Helper function to convert time string M:SS to total seconds
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

# Load and preprocess fight data
try:
    fight_data_path = os.path.join('new_fight_detail_full.csv')
    fight_data = pd.read_csv(fight_data_path)

    # Convert 'Time' from M:SS to seconds
    fight_data['Time'] = fight_data['Time'].apply(time_to_seconds)

    # Apply one-hot encoding to categorical columns that affect the model's decision
    fight_data = pd.get_dummies(fight_data, columns=['Weight Class', 'Winning Method', 'Win/Loss (Fighter1)'])

    # Optionally encode other categorical columns as needed using LabelEncoder or additional one-hot encoding
    encoder = LabelEncoder()
    fight_data['Other_Categorical_Column'] = encoder.fit_transform(fight_data['Other_Categorical_Column'])

except Exception as e:
    st.error(f"Failed to load or prepare fight data: {str(e)}")

# App title
st.title('Fight Win Predictor')

# Example of using the data for prediction
if st.button('Predict Outcome'):
    try:
        # Assuming the necessary preprocessing is done and you've chosen the features used in the model
        features = ['Time', 'Total_Fights'] + [col for col in fight_data.columns if 'Weight Class_' in col] + ...
        input_data = fight_data[features].iloc[0].values.reshape(1, -1)  # Reshape data for prediction

        # Perform prediction
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction[0] == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
