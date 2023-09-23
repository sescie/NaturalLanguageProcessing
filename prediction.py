#Imports
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the model and tokenizer

model = load_model('model2.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))



# generate a sequence from a language model
def generate_seq(model, tokenizer,seed_text):
        seq_length = 5
        n_words = 1
        result = list()
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
                # encode the text as integer
                encoded = tokenizer.texts_to_sequences([in_text])[0]
                # truncate sequences to a fixed length
                encoded = pad_sequences([encoded], maxlen=seq_length, truncating= 'pre' )
                # predict probabilities for each word
                #yhat = np.round(model.predict(encoded, verbose=0)).astype(int)
                yhat = np.argmax(model.predict(encoded, verbose=0),axis=1)
                # map predicted word index to word
                out_word = ''
                for word, index in tokenizer.word_index.items():
                        if index == yhat:
                                out_word = word
                                break
                # append to input
                in_text += ' ' + out_word
                result.append(in_text)
        return ' ' .join(result)



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
    seed_text = st.text_input("Enter any five words: ")
    lst = list(seed_text.split())

            

    if st.button("Generate"):
        
        if (seed_text is not None and len(lst)==5):
        
            result = generate_seq(model, tokenizer, seed_text)
            st.success(result)
        
        else:
            st.write("Please enter five words only")
        
        


if __name__ == '__main__':
    main()
