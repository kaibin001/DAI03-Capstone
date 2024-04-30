import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join('fight_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load fighter names and stats
file_3_data_path = os.path.join('File 3.csv')
fighter_names = pd.read_csv(file_3_data_path)

# Check and handle missing columns
expected_columns = ['Full Name', 'Win Rate', 'Total Fight', 'Win', 'Lose', 'Draw', 'Height', 'Weight', 'SLpM', 'Str Acc', 'SApM', 'Str Def', 'TD Avg', 'TD Acc', 'TD Def', 'Sub. Avg']
missing_columns = [col for col in expected_columns if col not in fighter_names.columns]
if missing_columns:
    st.error(f"Missing expected columns: {missing_columns}")
    st.stop()

# Convert percentage and ensure numeric
for col in ['Win Rate', 'Str Acc', 'SApM', 'Str Def', 'TD Avg', 'TD Acc', 'TD Def']:
    fighter_names[col] = pd.to_numeric(fighter_names[col].str.replace('%', ''), errors='coerce') / 100

# App title
st.title('Fight Win Predictor')

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Fighter 1', options=['Please select a fighter'] + sorted(fighter_names['Full Name'].tolist()), key='f1')
    if fighter1 != 'Please select a fighter':
        fighter1_stats = fighter_names[fighter_names['Full Name'] == fighter1]
        general_stats_1 = fighter1_stats[['Win Rate', 'Total Fight', 'Win', 'Lose', 'Draw', 'Height', 'Weight']]
        performance_stats_1 = fighter1_stats[['SLpM', 'Str Acc', 'SApM', 'Str Def', 'TD Avg', 'TD Acc', 'TD Def', 'Sub. Avg']]
        # Split the stats into two side-by-side columns
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.write('Fighter 1 Details:', general_stats_1)
        with stat_col2:
            st.write('Overall Stats', performance_stats_1)

with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=['Please select a fighter'] + sorted(fighter_names[fighter_names['Full Name'] != fighter1]['Full Name'].tolist()), key='f2')
    if fighter2 != 'Please select a fighter':
        fighter2_stats = fighter_names[fighter_names['Full Name'] == fighter2]
        general_stats_2 = fighter2_stats[['Win Rate', 'Total Fight', 'Win', 'Lose', 'Draw', 'Height', 'Weight']]
        performance_stats_2 = fighter2_stats[['SLpM', 'Str Acc', 'SApM', 'Str Def', 'TD Avg', 'TD Acc', 'TD Def', 'Sub. Avg']]
        # Split the stats into two side-by-side columns
        stat_col3, stat_col4 = st.columns(2)
        with stat_col3:
            st.write('Fighter 2 Details:', general_stats_2)
        with stat_col4:
            st.write('Overall Stats', performance_stats_2)

# Predict button
if st.button('Predict Outcome'):
    if fighter1 != 'Please select a fighter' and fighter2 != 'Please select a fighter':
        # Prepare data for prediction
        # Flatten and concatenate arrays, ensuring only numeric data is used
        f1_values = np.array(fighter1_stats.select_dtypes(include=[np.number]))
        f2_values = np.array(fighter2_stats.select_dtypes(include=[np.number]))
        input_data = np.concatenate([f1_values, f2_values]).reshape(1, -1)
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    else:
        st.error("Please select both fighters.")
