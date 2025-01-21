import json
import requests
from PIL import Image
import streamlit as st
import pickle
import numpy as np
#import streamlit_lottie as st_lottie

# Set page configuration
st.set_page_config(page_title="Prediction of Disease Outbreaks",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Define your custom CSS 
background_image_url = "https://png.pngtree.com/thumb_back/fw800/background/20230416/pngtree-medical-light-blue-background-image_2371492.jpg" 
page_bg_img = f""" 
<style> [data-testid="stAppViewContainer"] > .main {{ background-image: url("{background_image_url}"); 
background-size: cover; 
background-position: center; 
background-repeat: no-repeat; 
background-attachment: fixed; }} [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }} </style> 
"""
# Inject the css 
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the models
diabetes_model = pickle.load(open('D:\AICTE project\saved_models/diabetes_disease_model.sav', 'rb'))
heart_model = pickle.load(open('D:\AICTE project\saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('D:\AICTE project\saved_models/parkinsons_disease_model.sav', 'rb'))

# Create a function to get user input
def get_user_input(columns, prefix):
    user_input = []
    col1,col2=st.columns(2)
    for idx, col in enumerate(columns):
        with col1 if idx % 2 ==0 else col2:
            value = st.number_input(f'Enter value for {col}', value=0.0, key=f'{prefix}_{idx}')
            user_input.append(value)
    return np.array(user_input).reshape(1, -1)

# Descriptions for each disease 
descriptions = { 
    'diabetes': 'Positive prediction for diabetes indicates that the individual might have a high blood sugar level. It is advisable to consult with a healthcare professional for further assessment and management.', 
    'heart': 'Positive prediction for heart disease indicates that the individual might be at risk for cardiovascular issues. Immediate consultation with a healthcare provider is recommended for detailed evaluation.',
    'parkinsons': 'Positive prediction for Parkinson\'s disease suggests potential issues with motor control and neurological functions. Consulting a neurologist for a comprehensive diagnosis and management plan is essential.'
     }

#def load_lottiefile(filepath: str):
 #   with open(filepath, "r") as f:
  #      return json.load(f)
#lottie_anime= load_lottiefile(r"D:\AICTE project\anime.json")
#def load_lottieurl(url: str):
   # r= requests.get(url)
  #  if r.status_code != 200:
  #      return None
   # return r.json()
#lottie_doctor= load_lottieurl("https://lottie.host/cb695c3b-5159-4a57-a54c-e66ffa915840/1PFOydCHl0.json")
img=Image.open("doctor1.png")

# Title and animation side by side 
col1,col2= st.columns(2)
with col1:
    st.title('Disease Outbreak Prediction') 
with col2:
    st.image(
        img,width=300)



# Tabs for each disease
tab1, tab2, tab3 = st.tabs(['Diabetes', 'Heart Disease', 'Parkinson\'s'])

# Diabetes Tab
with tab1:
    st.header('Diabetes Prediction')
    diabetes_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_input = get_user_input(diabetes_columns,'diabetes')
    if st.button('Predict Diabetes'):
        prediction = diabetes_model.predict(diabetes_input)
        result = "Positive" if prediction[0] == 1 else "Negative" 
        st.write(f'Prediction: {result}') 
        if result == "Positive": 
            st.write(descriptions['diabetes'])

# Heart Disease Tab
with tab2:
    st.header('Heart Disease Prediction')
    heart_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Thallium', 'NumMajorVessels']
    heart_input = get_user_input(heart_columns,'heart')
    if st.button('Predict Heart Disease'):
        prediction = heart_model.predict(heart_input)
        result = "Positive" if prediction[0] == 1 else "Negative" 
        st.write(f'Prediction: {result}') 
        if result == "Positive": 
            st.write(descriptions['heart'])

# Parkinson's Tab
with tab3:
    st.header('Parkinson\'s Disease Prediction')
    parkinsons_columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    parkinsons_input = get_user_input(parkinsons_columns,'parkinsons')
    if st.button('Predict Parkinson\'s'):
        prediction = parkinsons_model.predict(parkinsons_input)
        result = "Positive" if prediction[0] == 1 else "Negative" 
        st.write(f'Prediction: {result}') 
        if result == "Positive": 
            st.write(descriptions['parkinsons'])


