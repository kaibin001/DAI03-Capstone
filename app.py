import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = os.path.join('fight_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load fight data and fighter names
fight_data_path = os.path.join('new_fight_detail_full.csv')
fight_data = pd.read_csv(fight_data_path)
file_3_data_path = os.path.join('File 3.csv')
fighter_names = pd.read_csv(file_3_data_path)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Combine all fighters into one series to fit the LabelEncoder
all_fighters = pd.concat([fight_data['Fighter1'], fight_data['Fighter2']]).unique()
label_encoder.fit(all_fighters)

# Encode Fighter1 and Fighter2 columns
fight_data['Fighter1'] = label_encoder.transform(fight_data['Fighter1'])
fight_data['Fighter2'] = label_encoder.transform(fight_data['Fighter2'])

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
    fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters, key='f1', format_func=lambda x: label_encoder.inverse_transform([x])[0])
    fighter1_stats = fight_data[fight_data['Fighter1'] == fighter1].iloc[0]
    st.write('Fighter 1 Stats:', fighter1_stats)
with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=[f for f in filtered_fighters if f != fighter1], key='f2', format_func=lambda x: label_encoder.inverse_transform([x])[0])
    fighter2_stats = fight_data[fight_data['Fighter1'] == fighter2].iloc[0]
    st.write('Fighter 2 Stats:', fighter2_stats)

# Predict button
if st.button('Predict Outcome'):
    try:
        # Assuming fighter1_stats and fighter2_stats are pandas Series with the same structure
        # Combine the relevant features from both fighters into a single array
        # Ensure there are no missing values or handle them appropriately
        fighter1_features = fighter1_stats.values[:-1]  # Adjust the slicing as per your data
        fighter2_features = fighter2_stats.values[:-1]  # Adjust the slicing as per your data
        input_data = np.array([np.concatenate((fighter1_features, fighter2_features))])

        # Perform prediction
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
