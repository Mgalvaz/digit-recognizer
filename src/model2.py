import streamlit as st
from model_view import render_model_page

st.title('Model 2')
render_model_page('model2', 'models/mnist_cnn2.keras', 'models/history2.json')