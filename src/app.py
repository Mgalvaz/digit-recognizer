import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.nn import softmax

model1 = load_model("models/mnist_cnn.keras")
model2 = load_model("models/mnist_cnn1.keras")

st.set_page_config(layout="wide")
st.title("Digit Recognizer Ensemble")

# -------------------
# Configuración Canvas
# -------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas"
if "pred_digit" not in st.session_state:
    st.session_state.pred_digit = None

# Botón para borrar
if st.button("Borrar dibujo"):
    st.session_state.canvas_key = "canvas_" + str(np.random.randint(10000))
    st.session_state.pred_digit = None

# Crear canvas
canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        display_toolbar= False
    )
# -------------------
# Predicción
# -------------------
threshold = 0.4  # Umbral de confianza
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Convertir canvas a array 28x28
        img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0]))
        img = img.resize((28, 28)).convert("L")
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(-1, 28, 28, 1).astype("float32")

        # Ensemble
        pred1 = softmax(model1.predict(img_array)).numpy()[0]
        pred2 = softmax(model2.predict(img_array)).numpy()[0]
        combined_pred = np.sqrt(np.power(pred1, 2) + np.power(pred2, 2))

        max_prob = np.max(combined_pred)
        digit = np.argmax(combined_pred)
        st.session_state.pred_digit = digit if max_prob >= threshold else None

        # -------------------
        # Mostrar predicción
        # -------------------
        if st.session_state.pred_digit is None and canvas_result.image_data is not None:
            st.write("No se ha reconocido un dígito con suficiente confianza.")
        elif st.session_state.pred_digit is not None:
            st.write(f"Número predicho: {st.session_state.pred_digit}, confianza: {np.max(combined_pred):.2f}")

        # -------------------
        # Visualización con Altair
        # -------------------
        if canvas_result.image_data is not None:
            digits = list(range(10))

            # DataFrame para barras por modelo
            df_models = pd.DataFrame({
                'Digit': digits * 2,
                'Probability': np.concatenate([pred1, pred2]),
                'Model': ['Model 1'] * 10 + ['Model 2'] * 10
            })

            # DataFrame para barras combinadas
            df_combined = pd.DataFrame({
                'Digit': digits,
                'Probability': combined_pred
            })

            # Crear columnas
            col1, col2 = st.columns(2)

            # Gráfico barras por modelo
            bars = alt.Chart(df_models).mark_bar().encode(
                x=alt.X('Digit:N', title='Digit'),
                y=alt.Y('Probability:Q', title='Probability'),
                color='Model:N',
                xOffset='Model:N'
            ).properties(width=30, height=300)

            threshold_line = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(
                color='red',
                strokeDash=[5, 5],
                size=2
            ).encode(y='y:Q')

            chart_models = bars + threshold_line
            col1.altair_chart(chart_models, use_container_width=True)

            # Gráfico barras combinadas
            bars_combined = alt.Chart(df_combined).mark_bar(color='purple').encode(
                x=alt.X('Digit:N', title='Digit'),
                y=alt.Y('Probability:Q', title='Probability')
            ).properties(width=30, height=300)

            threshold_line_combined = alt.Chart(pd.DataFrame({'y': [threshold]})).mark_rule(
                color='red',
                strokeDash=[5, 5],
                size=2
            ).encode(y='y:Q')

            chart_combined = bars_combined + threshold_line_combined
            col2.altair_chart(chart_combined, use_container_width=True)

