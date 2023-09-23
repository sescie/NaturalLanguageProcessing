#Imports
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('best_model2.h5',compile=False)
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))







def main():

    
    st.title("NLP: Language Modelling")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Text Generation App</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title("Generate the next word")
    user_input = st.text_input("Enter any five words: ")
    lst = list(seed_text.split())

            

    if st.button("Generate"):
        
        if (seed_text is not None and len(lst)==5):
        
            #result =  predict_next_words(model, tokenizer, user_input, num_words=3)
            st.success(result)
        
        else:
            st.write("Please enter five words only")
        
        


if __name__ == '__main__':
    main()
