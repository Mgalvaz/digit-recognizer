import streamlit as st
from model_view import render_model_page

st.title('Model 1')
render_model_page('model1', 'models/mnist_cnn1.keras', 'models/history1.json')