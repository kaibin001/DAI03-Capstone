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

# App title
st.title('Fight Win Predictor')

# Helper function to format stats tables
def format_stats(stats):
    # Convert relevant fields to float before formatting
    stats['Win Rate'] = float(stats['Win Rate'])
    stats['Str. Acc.'] = float(stats['Str. Acc.'])
    stats['Str. Def'] = float(stats['Str. Def'])
    stats['TD Acc.'] = float(stats['TD Acc.'])
    stats['TD Def.'] = float(stats['TD Def.'])
    
    # Apply percentage formatting
    stats['Win Rate'] = f"{stats['Win Rate']*100:.2f}%"
    stats['Str. Acc.'] = f"{stats['Str. Acc.']*100:.2f}%"
    stats['Str. Def'] = f"{stats['Str. Def']*100:.2f}%"
    stats['TD Acc.'] = f"{stats['TD Acc.']*100:.2f}%"
    stats['TD Def.'] = f"{stats['TD Def']*100:.2f}%"
    return stats

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Fighter 1', options=['Please select a fighter'] + sorted(fighter_names['Full Name'].tolist()), key='f1')
    if fighter1 != 'Please select a fighter':
        fighter1_stats = fighter_names[fighter_names['Full Name'] == fighter1].iloc[0]
        general_stats_1 = fighter1_stats[['Win Rate', 'Total Fight', 'Win', 'Lose', 'Draw', 'Height', 'Weight']]
        performance_stats_1 = fighter1_stats[['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']]
        st.write('Fighter 1 General Stats:', format_stats(general_stats_1))
        st.write('Fighter 1 Performance Stats:', format_stats(performance_stats_1))
with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=['Please select a fighter'] + sorted(fighter_names[fighter_names['Full Name'] != fighter1]['Full Name'].tolist()), key='f2')
    if fighter2 != 'Please select a fighter':
        fighter2_stats = fighter_names[fighter_names['Full Name'] == fighter2].iloc[0]
        general_stats_2 = fighter2_stats[['Win Rate', 'Total Fight', 'Win', 'Lose', 'Draw', 'Height', 'Weight']]
        performance_stats_2 = fighter2_stats[['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']]
        st.write('Fighter 2 General Stats:', format_stats(general_stats_2))
        st.write('Fighter 2 Performance Stats:', format_stats(performance_stats_2))

# Predict button
if st.button('Predict Outcome'):
    if fighter1 != 'Please select a fighter' and fighter2 != 'Please select a fighter':
        # Prepare data for prediction
        # Note: You'll need to properly prepare the feature vector here based on how your model was trained
        input_data = np.array([fighter1_stats.values[:-1] + fighter2_stats.values[:-1]])  # Adjust as per actual feature requirements
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    else:
        st.error("Please select both fighters.")
