from functools import partial
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.nn import softmax

MODEL1 = load_model(f'models/mnist_cnn1.keras')
MODEL2 = load_model(f'models/mnist_cnn2.keras')

def single_model(m1: bool):
    model = MODEL1 if m1 else MODEL2
    st.title('Digit Recognizer')

    # Canvas configuration
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas"
    canvas_result = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )

    #Prediction
    threshold = st.sidebar.number_input('Threshold: ', 0.0, 1.0, value = 0.4, disabled=True)
    if st.button("Predecir"):
        if canvas_result.image_data is not None:

            img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0]))
            img = img.resize((28, 28)).convert("L")
            img = ImageOps.invert(img)
            img_array = np.array(img).reshape(-1, 28, 28, 1).astype("float32")

            pred = softmax(model.predict(img_array)).numpy()[0]
            digit = np.argmax(pred)
            st.session_state.pred_digit = digit if pred[digit] >= threshold else None

            if st.session_state.pred_digit is None:
                st.write("There hasn't been a digit predicted with enough confidence..")
            else:
                st.write(f"Predicted number: {st.session_state.pred_digit}, confidence: {np.max(pred):.2f}")

            #Create chart
            digits = list(range(10))
            df = pd.DataFrame({
                'Digit': digits,
                'Probability': pred
            })
            bar = alt.Chart(df).mark_bar().encode(
                x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Probability:Q', title='Probability')
            ).properties(width=10, height=300)

            threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(
                color='red',
                strokeDash=[5, 5],
                size=2
            ).encode(y='y:Q')

            chart = bar + threshold_line
            st.altair_chart(chart)

def ensemble_model(arith: bool):
    model1 = MODEL1
    model2 = MODEL2

    st.title('Digit Recognizer')

    # --- Inicialización ---
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.5
    if "beta" not in st.session_state:
        st.session_state.beta = 0.5

    # --- Funciones de sincronización ---
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

    # --- Widgets ---
    st.sidebar.slider("Weighted parameters (Model 1)", 0.11, 0.99, value=st.session_state.alpha, key="alpha_slider", on_change=update_from_slider)
    c1, c2 = st.sidebar.columns(2)
    c1.number_input("Model 1:", 0.01, 0.99, value=st.session_state.alpha, key="alpha_num", on_change=update_from_alpha_num)
    c2.number_input("Model 2:", 0.01, 0.99, value=st.session_state.beta, key="beta_num", on_change=update_from_beta_num)

    # Valores finales sincronizados
    alpha = st.session_state.alpha
    beta = st.session_state.beta

    # Canvas configuration
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas"
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    # Prediction
    threshold = st.sidebar.number_input('Threshold: ', 0.0, 1.0, value = 0.4, disabled=True)
    if st.button("Predecir"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0]))
            img = img.resize((28, 28)).convert("L")
            img = ImageOps.invert(img)
            img_array = np.array(img).reshape(-1, 28, 28, 1).astype("float32")

            pred1 = softmax(model1.predict(img_array)).numpy()[0]
            pred2 = softmax(model2.predict(img_array)).numpy()[0]

            #Ensemble
            if arith:
                combined_pred = alpha * pred1 + beta * pred2
            else:
                combined_pred = np.power(pred1, alpha) * np.power(pred2, beta)
                combined_pred /= sum(combined_pred)

            digit = np.argmax(combined_pred)
            st.session_state.pred_digit = digit if combined_pred[digit] >= threshold else None
            if st.session_state.pred_digit is None:
                st.write("There hasn't been a digit predicted with enough confidence..")
            else:
                st.write(f"Predicted number: {st.session_state.pred_digit}, confidence: {np.max(combined_pred):.2f}")

            #Create charts
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

            threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(
                color='red',
                strokeDash=[5, 5],
                size=2
            ).encode(y='y:Q')
            bars = alt.Chart(df_models).mark_bar().encode(
                x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Probability:Q', title='Probability'),
                color='Model:N',
                xOffset='Model:N'
            ).properties(width=30, height=300)
            bars_combined = alt.Chart(df_combined).mark_bar(color='purple').encode(
                x=alt.X('Digit:N', title='Digit', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Probability:Q', title='Probability')
            ).properties(width=30, height=300)

            col1, col2 = st.columns(2)
            chart_models = bars + threshold_line
            chart_combined = bars_combined + threshold_line
            col1.altair_chart(chart_models, use_container_width=True)
            col2.altair_chart(chart_combined, use_container_width=True)


models = {
    'Model 1': partial(single_model, True),
    'Model 2': partial(single_model, False),
    'Arithmetic Mean': partial(ensemble_model, True),
    'Geometric Mean': partial(ensemble_model, False)
}

st.sidebar.header('Model Configuration')
model_type = st.sidebar.selectbox('Model:', options=models.keys())
st.set_page_config(page_title='App Digit Recognizer')
models[model_type]()
with st.sidebar:
    st.markdown('---')

