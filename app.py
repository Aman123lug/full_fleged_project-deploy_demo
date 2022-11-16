
import streamlit as st
from flask import Flask, app, jsonify, url_for, render_template

from tensorflow import keras
model = keras.models.load_model('my_model.h5')

st.title("Our Project :smile:")

