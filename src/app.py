import streamlit as st
import os

pages = [
    st.Page('recognizer.py', title='Digit recognizer'),
    st.Page('model1.py', title='About model 1'),
    st.Page('model2.py', title='About model 2')
]
st.set_page_config(page_title='MNIST Digit Recognizer')
pg = st.navigation(pages, position='top')

# Check for missing models
model_files = {'model1': 'models/mnist_cnn1.keras', 'model2': 'models/mnist_cnn2.keras'}
missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]
if missing_models:
    st.error(
        f"The following models have not been found: {', '.join(missing_models)}.\n\n"
        f"Execute the `train_model.py` script before launching the app."
    )
    st.stop()

pg.run()