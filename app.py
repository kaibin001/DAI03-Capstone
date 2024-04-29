{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "00a15aa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import os\n",
    "\n",
    "# Load the trained model\n",
    "home_directory = os.path.expanduser('~')\n",
    "model_path = os.path.join(home_directory, 'Downloads/Capstone/fight_model.pkl')\n",
    "with open(model_path, 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Load fight data and fighter names\n",
    "fight_data_path = os.path.join(home_directory, 'Downloads/Capstone/new_fight_detail_full.csv')\n",
    "fight_data = pd.read_csv(fight_data_path)\n",
    "file_3_data_path = os.path.join(home_directory, 'Downloads/Capstone/File 3.csv')\n",
    "fighter_names = pd.read_csv(file_3_data_path)\n",
    "\n",
    "# Convert all columns to string to avoid serialization issues\n",
    "fighter_names = fighter_names.astype(str)\n",
    "fight_data = fight_data.astype(str)\n",
    "\n",
    "# App title\n",
    "st.title('Fight Win Predictor')\n",
    "\n",
    "# Weight class selection\n",
    "weight_classes = fight_data['Weight Class'].unique()\n",
    "selected_weight_class = st.selectbox('Select Weight Class', weight_classes)\n",
    "\n",
    "# Filter fighters based on selected weight class\n",
    "filtered_fighters = fight_data[fight_data['Weight Class'] == selected_weight_class]['Fighter1'].unique()\n",
    "\n",
    "# Fighter dropdowns\n",
    "col1, col2 = st.columns(2)\n",
    "with col1:\n",
    "    fighter1 = st.selectbox('Select Fighter 1', options=filtered_fighters, key='f1')\n",
    "    fighter1_stats = fight_data[fight_data['Fighter1'] == fighter1].iloc[0]\n",
    "    st.write('Fighter 1 Stats:', fighter1_stats)\n",
    "with col2:\n",
    "    fighter2 = st.selectbox('Select Fighter 2', options=[f for f in filtered_fighters if f != fighter1], key='f2')\n",
    "    fighter2_stats = fight_data[fight_data['Fighter1'] == fighter2].iloc[0]\n",
    "    st.write('Fighter 2 Stats:', fighter2_stats)\n",
    "\n",
    "# Predict button\n",
    "if st.button('Predict Outcome'):\n",
    "    # Prepare data for prediction (adjust this according to your model's trained features)\n",
    "    input_data = np.array([fighter1_stats.values[:-1] + fighter2_stats.values[:-1]])  # Adjust as per actual feature requirements\n",
    "    prediction = model.predict(input_data)\n",
    "    win_status = 'Fighter 1 Wins' if prediction == 1 else 'Fighter 2 Wins'\n",
    "    st.success(f'Prediction: {win_status}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
