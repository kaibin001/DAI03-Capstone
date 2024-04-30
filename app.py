import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Load data and model

data_path = os.path.join('File 3.csv')
model_file = os.path.join('new_fight_detail_full.csv')
model_path = os.path.join('fight_model.pkl')

fighters_data = pd.read_csv(model_file)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Ensure numeric data and handle NaNs
cols_to_numeric = ['Win', 'Lose', 'Draw']
fighters_data[cols_to_numeric] = fighters_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate additional stats
fighters_data['Total Fights'] = fighters_data['Win'] + fighters_data['Lose'] + fighters_data['Draw']
fighters_data['Win Rate'] = fighters_data['Win'] / fighters_data['Total Fights']

# Handle division by zero if Total Fights is zero
fighters_data['Win Rate'] = fighters_data['Win Rate'].fillna(0)

# Streamlit layout
st.title("UFC Fight Prediction")

# Dropdown for Fighter 1
fighter_1 = st.selectbox("Fighter 1", ["Please select or type in the Fighter's name"] + fighters_data['Full Name'].tolist())
filtered_data = fighters_data[fighters_data['Full Name'] != fighter_1]

# Dropdown for Fighter 2
fighter_2 = st.selectbox("Fighter 2", ["Please select or type in the Fighter's name"] + filtered_data['Full Name'].tolist())

# Show fighter stats if selected
if fighter_1 != "Please select or type in the Fighter's name":
    st.write("Fighter 1 Stats:")
    st.write(fighters_data[fighters_data['Full Name'] == fighter_1][['Total Fights', 'Win Rate', 'Win', 'Lose', 'Draw', 'Height', 'Weight', 'SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']])

if fighter_2 != "Please select or type in the Fighter's name":
    st.write("Fighter 2 Stats:")
    st.write(fighters_data[fighters_data['Full Name'] == fighter_2][['Total Fights', 'Win Rate', 'Win', 'Lose', 'Draw', 'Height', 'Weight', 'SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']])

# Prediction button and results
if st.button("Predict Fight Outcome"):
    # Ensure both fighters are selected
    if fighter_1 != "Please select or type in the Fighter's name" and fighter_2 != "Please select or type in the Fighter's name":
        # Prepare features and scale them if necessary
        # Here you need to implement the exact preprocessing required by your model
        # For example, this might include extracting and scaling the difference between fighters' stats
        # Ensure that preprocessing steps match those used during model training
        st.write("Prediction result would be displayed here.")  # Replace this with your prediction code
    else:
        st.error("Please select fighters for both Fighter 1 and Fighter 2.")
