import pandas as pd
import numpy as np
import streamlit as st
from itertools import chain
from flask import Flask, app, jsonify, url_for, render_template
from keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras
model = keras.models.load_model('my_model.h5')
data = pd.read_csv("/home/user/Downloads/data.csv")[0:1000]
data.head()
X = data["Headline"]
st.title("Our Project :smile:")

text = st.text_input("enter text")
# text = "Egypt's Cheiron wins tie-up    with Pemex for Mexican onshore oil field"
text = text.zfill(110)
token = Tokenizer()
token.fit_on_texts(X)
seq = token.texts_to_sequences(text)
final_input = list(chain.from_iterable(seq))
# a = [i for i in final_input ]
for i in range(len(final_input)):
    if final_input[i] == 431:
        final_input[i] = 0
        




final_input = np.array(final_input)
print(type(final_input))
y = model.predict(final_input)
print(y)

