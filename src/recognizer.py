import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.nn import softmax
from functools import partial

THRESHOLD = 0.7 # Change this number if you want a different threshold. It should be in the range (0,1)

# Reads the image drawn in the canvas and converts it to an array that can be fed to the CNN
def process_image(image_data):
    img = Image.fromarray(np.uint8(image_data[:, :, 0]))
    img = img.resize((28, 28)).convert('L')
    img = ImageOps.invert(img)
    return np.asarray(img).reshape(-1, 28, 28, 1).astype('float32')

# Supage for single model option
def single_model(model_key: str):
    model = st.session_state[model_key]
    title = alt.TitleParams('Model 1' if model_key=='model1' else 'Model 2', anchor='middle')

    # Prediction
    thr = st.sidebar.number_input('Threshold: ', 0.0, 1.0, value = THRESHOLD, disabled=True)
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            img_array = process_image(canvas_result.image_data)
            with st.spinner('Predicting'):

                # Digit prediction
                pred = softmax(model.predict(img_array)).numpy()[0]
                digit = np.argmax(pred)
                if pred[digit] < thr:
                    st.write("There hasn't been a digit predicted with enough confidence.")
                else:
                    st.write(f'Predicted number: {digit}, confidence: {pred[digit]:.2f}')

                # Chart of digit probabilities
                digits = list(range(10))
                df = pd.DataFrame({
                    'Digit': digits,
                    'Probability': pred
                })
                bar = alt.Chart(df, title=title).mark_bar().encode(
                    x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Probability:Q', title='Probability')
                ).properties(width=10, height=300, title=title)
                threshold_line = alt.Chart(pd.DataFrame({'y': [thr]})).mark_rule(
                    color='red',
                    strokeDash=[5, 5],
                    size=2
                ).encode(y='y:Q')
                chart = bar + threshold_line
                st.altair_chart(chart)

# Subpage for arithmetic or geometric ensemble option
def ensemble_model(arith: bool):
    model1 = st.session_state.model1
    model2 = st.session_state.model2

    # Update functions for sidebar and number inputs
    def update_from_slider():
        st.session_state.alpha = st.session_state.alpha_slider
        st.session_state.beta = 1 - st.session_state.alpha_slider
        st.session_state.alpha_num = st.session_state.alpha
        st.session_state.beta_num = st.session_state.beta
    def update_from_alpha_num():
        st.session_state.alpha = st.session_state.alpha_num
        st.session_state.beta = 1 - st.session_state.alpha_num
        st.session_state.alpha_slider = st.session_state.alpha
        st.session_state.beta_num = st.session_state.beta
    def update_from_beta_num():
        st.session_state.beta = st.session_state.beta_num
        st.session_state.alpha = 1 - st.session_state.beta_num
        st.session_state.alpha_num = st.session_state.alpha
        st.session_state.alpha_slider = st.session_state.alpha

    # Selection of alpha (weight of model 1) and beta (weight of model 2)
    st.sidebar.slider('Weighted parameters (Model 1)', 0.0, 1.0, value=st.session_state.alpha, key='alpha_slider', on_change=update_from_slider)
    c1, c2 = st.sidebar.columns(2)
    c1.number_input('Model 1:', 0.0, 1.0, value=st.session_state.alpha, key='alpha_num', on_change=update_from_alpha_num)
    c2.number_input('Model 2:', 0.0, 1.0, value=st.session_state.beta, key='beta_num', on_change=update_from_beta_num)
    alpha = st.session_state.alpha
    beta = st.session_state.beta

    # Prediction
    thr = st.sidebar.number_input('Threshold: ', 0.0, 1.0, value = THRESHOLD, disabled=True)
    if st.button('Predict'):
        if canvas_result.image_data is not None:
            # Digit prediction
            img_array = process_image(canvas_result.image_data)
            with st.spinner('Predicting'):
                pred1 = softmax(model1.predict(img_array)).numpy()[0]
                pred2 = softmax(model2.predict(img_array)).numpy()[0]

                # Ensemble
                if arith:
                    combined_pred = alpha * pred1 + beta * pred2
                else:
                    combined_pred = np.power(pred1, alpha) * np.power(pred2, beta)
                    combined_pred /= sum(combined_pred)
                digit = np.argmax(combined_pred)
                if combined_pred[digit] < thr:
                    st.write("There hasn't been a digit predicted with enough confidence.")
                else:
                    st.write(f'Predicted number: {digit}, confidence: {combined_pred[digit]:.2f}')

                # Create charts of digit probabilities for each model and for the ensemble one
                digits = list(range(10))
                df_models = pd.DataFrame({
                    'Digit': digits * 2,
                    'Probability': np.concatenate([pred1, pred2]),
                    'Model': ['Model 1'] * 10 + ['Model 2'] * 10
                })
                df_combined = pd.DataFrame({
                    'Digit': digits,
                    'Probability': combined_pred
                })
                title_models = alt.TitleParams('Models', anchor='middle')
                title_combined = alt.TitleParams('Ensemble', anchor='middle')
                threshold_line = alt.Chart(pd.DataFrame({'y': [thr]})).mark_rule(
                    color='red',
                    strokeDash=[5, 5],
                    size=2
                ).encode(y='y:Q')
                bars_models = alt.Chart(df_models, title=title_models).mark_bar().encode(
                    x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Probability:Q', title='Probability'),
                    color='Model:N',
                    xOffset='Model:N'
                ).properties(width=30, height=300)
                bars_combined = alt.Chart(df_combined, title=title_combined).mark_bar(color='purple').encode(
                    x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Probability:Q', title='Probability')
                ).properties(width=30, height=300)
                col1, col2 = st.columns(2)
                chart_models = bars_models + threshold_line
                chart_combined = bars_combined + threshold_line
                col1.altair_chart(chart_models, use_container_width=True)
                col2.altair_chart(chart_combined, use_container_width=True)

models = {
    'Model 1': partial(single_model, 'model1'),
    'Model 2': partial(single_model, 'model2'),
    'Arithmetic Mean': partial(ensemble_model, True),
    'Geometric Mean': partial(ensemble_model, False)
}

if 'model1' not in st.session_state:
    st.session_state.model1 = load_model(f'models/mnist_cnn1.keras')
if 'model2' not in st.session_state:
    st.session_state.model2 = load_model(f'models/mnist_cnn2.keras')
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.5
if 'beta' not in st.session_state:
    st.session_state.beta = 0.5

st.title('Digit Recognizer')
st.sidebar.header('Model Configuration')
model_type = st.sidebar.selectbox('Model used to predict:', options=models.keys())
canvas_result = st_canvas(
        fill_color='white',
        stroke_width=10,
        stroke_color='black',
        background_color='white',
        width=280,
        height=280,
        drawing_mode='freedraw',
        key='canvas'
    )
models[model_type]()

with st.sidebar:
    st.markdown('---')
    st.write('Created using')
    st.image('https://www.tensorflow.org/images/tf_logo_social.png', width=120)

