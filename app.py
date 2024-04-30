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

# Load fight data
fight_data_path = os.path.join('new_fight_detail_full.csv')
fight_data = pd.read_csv(fight_data_path)

# Apply one-hot encoding to 'Weight Class'
fight_data = pd.get_dummies(fight_data, columns=['Weight Class'])

# Initialize a LabelEncoder and encode fighter names
label_encoder = LabelEncoder()
# Create a combined series of all fighters to ensure all names are included in the encoding
all_fighters = pd.concat([fight_data['Fighter1'], fight_data['Fighter2']]).unique()
label_encoder.fit(all_fighters)
fight_data['Fighter1'] = label_encoder.transform(fight_data['Fighter1'])
fight_data['Fighter2'] = label_encoder.transform(fight_data['Fighter2'])

# App title
st.title('Fight Win Predictor')

# Fighter dropdowns
col1, col2 = st.columns(2)
with col1:
    filtered_fighters = fight_data['Fighter1'].unique()
    fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters, format_func=lambda x: label_encoder.inverse_transform([x])[0])
    fighter1_stats = fight_data[fight_data['Fighter1'] == fighter1].iloc[0]
    st.write('Fighter 1 Stats:', fighter1_stats)
with col2:
    fighter2 = st.selectbox('Select Fighter 2', options=filtered_fighters, format_func=lambda x: label_encoder.inverse_transform([x])[0])
    fighter2_stats = fight_data[fight_data['Fighter2'] == fighter2].iloc[0]
    st.write('Fighter 2 Stats:', fighter2_stats)

# Predict button
if st.button('Predict Outcome'):
    try:
        # Assuming fighter1_stats and fighter2_stats are pandas Series with the same structure
        # Ensure there are no missing values or handle them appropriately
        # Combine the relevant features from both fighters into a single array
        fighter1_features = fight_data.loc[fight_data['Fighter1'] == fighter1].drop(['Fighter1', 'Fighter2'], axis=1).iloc[0].values
        fighter2_features = fight_data.loc[fight_data['Fighter2'] == fighter2].drop(['Fighter1', 'Fighter2'], axis=1).iloc[0].values
        input_data = np.array([np.concatenate((fighter1_features, fighter2_features))])
        
        # Perform prediction
        prediction = model.predict(input_data)
        win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'
        st.success(f'Prediction: {win_status}')
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

