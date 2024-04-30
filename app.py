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

# Assuming all prior data loading and processing (including encoding) are correctly done
# Define the model features according to the actual features used in training
model_features = [
    'Round', 'Total Fights (Fighter 1)', 'Win (Fighter 1)', 'Lose (Fighter 1)', 'Draw (Fighter 1)',
    'SLpM (Fighter 1)', 'Str. Acc. (Fighter 1)', 'SApM (Fighter 1)', 'Str. Def (Fighter 1)',
    'TD Avg. (Fighter 1)', 'TD Acc. (Fighter 1)', 'TD Def. (Fighter 1)', 'Sub. Avg. (Fighter 1)',
    'Total Fights (Fighter 2)', 'Win (Fighter 2)', 'Lose (Fighter 2)', 'Draw (Fighter 2)',
    'SLpM (Fighter 2)', 'Str. Acc. (Fighter 2)', 'SApM (Fighter 2)', 'Str. Def (Fighter 2)',
    'TD Avg. (Fighter 2)', 'TD Acc. (Fighter 2)', 'TD Def. (Fighter 2)', 'Sub. Avg. (Fighter 2)',
    'Weight Class'  # Ensure this is properly encoded if used as a feature
]

# Streamlit UI code
st.title('Fight Win Predictor')

# Selecting a weight class (after encoding)
weight_classes = fight_data['Weight Class'].unique()
selected_weight_class = st.selectbox('Select Weight Class', options=weight_classes, format_func=lambda x: label_encoders['Weight Class'].inverse_transform([x])[0])

# Filter fighters based on selected weight class
filtered_fight_data = fight_data[fight_data['Weight Class'] == selected_weight_class]
filtered_fighters1 = filtered_fight_data['Fighter1'].unique()
filtered_fighters2 = filtered_fight_data['Fighter2'].unique()

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters1, format_func=lambda x: label_encoders['Fighter1'].inverse_transform([x])[0])
    fighter1_stats = filtered_fight_data[filtered_fight_data['Fighter1'] == fighter1].iloc[0]
    st.write('Fighter 1 Stats:', fighter1_stats)
with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=filtered_fighters2, format_func=lambda x: label_encoders['Fighter2'].inverse_transform([x])[0])
    fighter2_stats = filtered_fight_data[filtered_fight_data['Fighter2'] == fighter2].iloc[0]
    st.write('Fighter 2 Stats:', fighter2_stats)

# Predict button
if st.button('Predict Outcome'):
    try:
        # Filter out to use only the required features
        fighter1_features = filtered_fight_data.loc[filtered_fight_data['Fighter1'] == fighter1, model_features].iloc[0].values
        fighter2_features = filtered_fight_data.loc[filtered_fight_data['Fighter2'] == fighter2, model_features].iloc[0].values
        input_data = np.array([np.concatenate((fighter1_features, fighter2_features))])

        # Perform prediction
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


