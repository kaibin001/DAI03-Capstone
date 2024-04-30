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

    # If other categorical columns need to be encoded, use LabelEncoder or additional one-hot encoding here
    # Example for a hypothetical column 'Other_Categorical_Column'
    # encoder = LabelEncoder()
    # fight_data['Other_Categorical_Column'] = encoder.fit_transform(fight_data['Other_Categorical_Column'])

except Exception as e:
    st.error(f"Failed to load or prepare fight data: {str(e)}")

# App title
st.title('Fight Win Predictor')

# Selection for weight class
try:
    weight_classes = fight_data['Weight Class'].unique()  # This assumes 'Weight Class' is still a valid column name
    selected_weight_class = st.selectbox('Select Weight Class', options=weight_classes)
    filtered_fight_data = fight_data[fight_data['Weight Class'] == selected_weight_class]
except Exception as e:
    st.error(f"Failed to setup weight class selection: {str(e)}")

# Fighter dropdowns
try:
    filtered_fighters = filtered_fight_data['Fighter1'].unique()  # Adjust according to how fighters are identified post-encoding
    col1, col2 = st.columns(2)
    with col1:
        fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters)
    with col2:
        fighter2 = st.selectbox('Select Fighter 2', options=filtered_fighters)
except Exception as e:
    st.error(f"Failed to setup fighter selection: {str(e)}")

# Predict button
if st.button('Predict Outcome'):
    try:
        # Prepare input data for prediction
        # Make sure to adjust feature selection to match the model's expectations
        features = ['feature1', 'feature2']  # Placeholder for actual feature names
        fighter1_data = filtered_fight_data[filtered_fight_data['Fighter1'] == fighter1][features].iloc[0].values
        fighter2_data = filtered_fight_data[filtered_fight_data['Fighter2'] == fighter2][features].iloc[0].values
        input_data = np.array([np.concatenate((fighter1_data, fighter2_data))])

        # Perform prediction
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
