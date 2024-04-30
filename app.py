import streamlit as st
import pandas as pd
import pickle

# Load the model and data
model = pickle.load(open('fight_model3.pkl', 'rb'))
fighter_details = pd.read_csv('File 3.csv')
fight_events = pd.read_csv('new_fight_detail_full.csv')


def process_features(fighter1, fighter2):
    # You need to replace this part with actual feature processing based on your model
    # This is just a placeholder
    features_f1 = fight_events.loc[fight_events['Fighter1']==fighter1]
    features_f1 = features_f1.iloc[:,[7,8,9,10,11,12,13,14,15,16,17,18,19]]
    # features_f1 = features_f1[features_f1.columns([4,5,7,8,9,10,11,12,13,14,15,16,17,18,19])]
    features_f1 = features_f1.iloc[0:1,:]
    features_f2 = fight_events.loc[fight_events['Fighter2']==fighter2]
    features_f2 = features_f2.iloc[:,[20,21,22,23,24,25,26,27,28,29,30,31,32]]
    # features_f2 = features_f2[features_f2.columns([4,5,20,21,22,23,24,25,26,27,28,29,30,31,32])]
    features_f2 = features_f2.iloc[0:1,:]
    # features = pd.merge(features_f1,features_f2, on='Weight Class')
    features_f1.reset_index(drop=True, inplace=True)
    features_f2.reset_index(drop=True, inplace=True)
    features = pd.concat([features_f1,features_f2], axis=1)
    
    percentage_features = ['Win Rate (Fighter 1)', 'Str. Acc. (Fighter 1)', 'Str. Def (Fighter 1)', 
                       'TD Acc. (Fighter 1)', 'TD Def. (Fighter 1)', 'Win Rate (Fighter 2)', 
                       'Str. Acc. (Fighter 2)', 'Str. Def (Fighter 2)', 'TD Acc. (Fighter 2)', 
                       'TD Def. (Fighter 2)']
    for feature in percentage_features:
        fight_data[feature] = fight_data[feature].str.rstrip('%').astype('float') / 100
    
    st.write(features_f1)
    st.write(features_f2)
    return features
    
# Application Title
st.title("UFC Fight Predictor")

# Fighter Selection
fighter_names = sorted(fighter_details['Full Name'].dropna().unique())
fighter1 = st.selectbox("Select Fighter 1", [''] + fighter_names)
if fighter1:
    st.write("Fighter 1 Stats", fight_events[fight_events['Fighter1'] == fighter1])
fighter2 = st.selectbox("Select Fighter 2", [''] + [f for f in fighter_names if f != fighter1])
if fighter2:
    st.write("Fighter 2 Stats", fight_events[fight_events['Fighter2'] == fighter2])


# Prediction Button
if st.button("Predict Winner"):
    if not fighter1 or not fighter2:
        st.error("Please select both fighters.")
    else:
        # Assuming you have a function to process input features
        input_features = process_features(fighter1, fighter2)
        st.write(fighter1)
        st.write(fighter2)
        st.write(input_features)
        prediction = model.predict([input_features])
        winner = "Fighter 1 Wins" if prediction == 1 else "Fighter 2 Wins"
        st.success(winner)


