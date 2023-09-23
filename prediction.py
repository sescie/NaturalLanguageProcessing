import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the previously saved model
model = load_model('best_model2.h5')

def predict_next_words(model, tokenizer, text, num_words=1):
    # ... (Your predict_next_words function as defined)

# Streamlit app
    st.title("Shona Text Prediction App")

# Input box for user to enter text
    user_input = st.text_input("Please type five words in Shona:")

    if user_input:
    # Predict the next words
       predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
       st.write(f"The next words might be: {predicted_words}")
