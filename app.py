import streamlit as st
import pandas as pd
import pickle

# Load the model and data
model = pickle.load(open('fight_model.pkl', 'rb'))
fighter_details = pd.read_csv('File 3.csv')
fight_events = pd.read_csv('new_fight_detail_full.csv')


def process_features(fighter1, fighter2):
    # You need to replace this part with actual feature processing based on your model
    # This is just a placeholder
    features_f1 = fight_events.loc[fight_events['Fighter1']==fighter1]
    features_f1 = features_f1.iloc[:,7:20]
    features_f2 = fight_events.loc[fight_events['Fighter2']==fighter2]
    features_f2 = features.iloc[:,20:]
    pd.concat([features_f1,features_f2], axis=1)
    return features
    
# Application Title
st.title("UFC Fight Predictor")

# Fighter Selection
fighter_names = sorted(fighter_details['Full Name'].dropna().unique())
fighter1 = st.selectbox("Select Fighter 1", [''] + fighter_names)
if fighter1:
    st.write("Fighter 1 Stats", fighter_details[fighter_details['Full Name'] == fighter1])
fighter2 = st.selectbox("Select Fighter 2", [''] + [f for f in fighter_names if f != fighter1])
if fighter2:
    st.write("Fighter 2 Stats", fighter_details[fighter_details['Full Name'] == fighter2])


# Prediction Button
if st.button("Predict Winner"):
    if not fighter1 or not fighter2:
        st.error("Please select both fighters.")
    else:
        # Assuming you have a function to process input features
        input_features = process_features(fighter1, fighter2)
        st.write(fighter1)
        st.write(fighter2)
        prediction = model.predict([input_features])
        winner = "Fighter 1 Wins" if prediction == 1 else "Fighter 2 Wins"
        st.success(winner)


