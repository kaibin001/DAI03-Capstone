import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Load data and model
home_directory = os.path.expanduser('~')
data_path = os.path.join(home_directory, 'File 3.csv')
model_path = os.path.join(home_directory, 'fight_model.pkl')

fighters_data = pd.read_csv(data_path)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Calculate additional stats
fighters_data['Total Fights'] = fighters_data['Win'] + fighters_data['Lose'] + fighters_data['Draw']
fighters_data['Win Rate'] = fighters_data['Win'] / fighters_data['Total Fights']

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
    # Prepare the data for prediction
    if fighter_1 != "Please select or type in the Fighter's name" and fighter_2 != "Please select or type in the Fighter's name":
        # Example of fetching features and preprocessing
        # This needs to be adapted to your specific feature set and preprocessing steps
        features_f1 = fighters_data[fighters_data['Full Name'] == fighter_1][['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']].values
        features_f2 = fighters_data[fighters_data['Full Name'] == fighter_2][['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', 'TD Def.', 'Sub. Avg.']].values
        
        # Assuming you need to normalize or scale features
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(features_f1 - features_f2)  # Example feature manipulation
        
        # Predict outcome
        prediction_probability = model.predict_proba(combined_features)[0]
        st.write(f"Probability of {fighter_1} winning: {prediction_probability[1]:.2f}")
        st.write(f"Probability of {fighter_2} winning: {1 - prediction_probability[1]:.2f}")
    else:
        st.error("Please select fighters for both Fighter 1 and Fighter 2.")

