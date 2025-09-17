import os

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import keras
from keras.models import load_model


MODEL_PATH = "models/MNIST-83%.h5"
#model = load_model(MODEL_PATH)

st.markdown("<h1 style='text-align:center'>MNIST Digit Recognizer<h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Canvas config
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar = False
    )

    if st.button("Predecir"):
        if canvas_result.image_data is not None:

            # Preprocess the image to make it 28x28x1 an invert it
            img = Image.fromarray(np.uint8(canvas_result.image_data[:, :, 0]))
            img = img.resize((28, 28)).convert("L")
            img = ImageOps.invert(img)
            img_array = np.array(img).reshape(-1, 28, 28, 1).astype("float32")


            # Predict the image
            #pred = model.predict(img_array)
            #digit = np.argmax(pred)
            #st.write(f"Número predicho: {digit}")
            #st.bar_chart(pred[0])
if __name__ == "__main__":
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "MNIST-83%.keras")
    model = keras.models.load_model(MODEL_PATH)

    print(model)
