import streamlit as st

pages = [
    st.Page('recognizer.py', title='Digit recognizer'),
    st.Page('model1.py', title='About model 1'),
    st.Page('model2.py', title='About model 2')
]
st.set_page_config(page_title='MNIST Digit Recognizer')
pg = st.navigation(pages, position='top')
pg.run()