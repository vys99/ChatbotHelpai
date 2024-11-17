import pickle
import streamlit as st
from datetime import date
import spacy
import random 
import numpy as np

# Load the chatbot model
with open(r"C:\Users\vysak\OneDrive\Desktop\python\helpai.pkl", 'rb') as f:
    model = pickle.load(f)

# Set up chatbot responses
date_tod = date.today()
responses = {
    'greetings': ['Hello! How can I help you today?', 'Hi there! What can I do for you?', 'Hi! Nice to see you here!'],
    'name': ['My name is HelpAI', 'I am HelpAI', 'I’m HelpAI, here to help you with any questions you have!', 'I’m HelpAI, nice to meet you! How can I help?'],
    'closing': ["Goodbye! Have a great day ahead!", "Take care! Looking forward to our next chat.", "Bye for now! Don’t hesitate to come back if you have more questions.", "See you soon! Wishing you all the best."],
    'date': [f'Today is {date_tod}', f'{date_tod}', f'Today\'s date is {date_tod}'],
    'wellness': ['I\'m doing great, thank you for asking!', "I'm good, how about you?", "I'm doing fine, thanks for checking in!"],
    'random': ["Oh! Do you know Octopuses have three hearts—two pump blood to the gills, while the third pumps it to the rest of the body.",
               "Ok. Here you go - The Eiffel Tower can grow taller in summer—the metal expands due to the heat, making the tower about 6 inches taller during hot weather.",
               "Why don't skeletons fight each other?....... Because they don't have the guts!  HA HA HA"],
    'owner': ["My Owner is Vysakh", "I was programmed by Vysakh", "My Owner's name is Vysakh", "His name is Vysakh"]
}

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocess user input
def preprocess(txt):
    doc = nlp(txt)
    inp_dat = doc.vector
    reshaped_data = np.reshape(inp_dat, (1, 96))
    return reshaped_data

# Predict response class
def pred(res):
    out_val = model.predict([res])
    return np.array(out_val)

# Generate chatbot response
def chatbot_response(user_input):
    res = preprocess(user_input)
    out_value = pred(res)
    res = np.argmax(out_value)
    thr = max(out_value[0])

    if thr > 0.9:
        if res == 0:
            return random.choice(responses['greetings'])
        elif res == 1:
            return random.choice(responses['name'])
        elif res == 2:
            return random.choice(responses['date'])
        elif res == 3:
            return random.choice(responses['closing'])
        elif res == 4:
            return random.choice(responses['wellness'])
        elif res == 5:
            return random.choice(responses['random'])
        elif res == 6:
            return random.choice(responses['owner'])
    else:
        return "I'm sorry, I didn't understand that. Can you rephrase?"

# Streamlit app
st.title("HelpAI Chatbot")

# Input widget for user messages
user_input = st.text_input("You:", placeholder="Type your message here...", key="user_input_key")

# Display chatbot response
if user_input:
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100, key="chatbot_response_key")
