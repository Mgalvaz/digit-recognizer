import json
import streamlit as st
from graphviz import Digraph
from keras.models import load_model
import altair as alt
import pandas as pd

def tuple_str(t):
    if len(t) == 1:
        return f'({t[0]})'
    return str(t)

st.title('Model 1')
if 'model1' not in st.session_state:
    st.session_state.model1 = load_model(f'models/mnist_cnn1.keras')
model = st.session_state.model1

#tab1, tab2, tab3 = st.tabs(["Model structure", "Trainig history", 'Test values'])
#Create diagram with the layers of the model
dot = Digraph(format="svg")
for i, layer in enumerate(model.layers):
    #Info of the layer
    input_shape = tuple_str(layer.input_shape[1:])
    output_shape = tuple_str(layer.output_shape[1:])
    params = layer.count_params()

    #Structure of each layer
    if params > 0:
        label = f'''{{ {layer.__class__.__name__} |
                {{ Input: {input_shape} | Output: {output_shape} }} |
                Params: {params} }}'''
    else:
        label = f'''{{ {layer.__class__.__name__} |
                {{ Input: {input_shape} | Output: {output_shape} }} }}'''

    #Create the layer
    dot.node(str(i), label=label, shape='record', style='filled', fillcolor='lightblue', tooltip= f'Layer {i+1}')

    #Connect it to the previous one
    if i > 0:
        dot.edge(str(i-1), str(i))

st.subheader('Model structure')
st.graphviz_chart(dot.source)

st.subheader('Training history')
with open('models/history1.json', 'r') as f:
    data = json.load(f)
st.line_chart(data['history'], x_label='epoch', y_label='value')

st.subheader('Test values')
df_test = pd.DataFrame.from_dict(data['test'], orient='index').reset_index()
df_test.columns = ['metric', 'value']

bar = alt.Chart(df_test).mark_bar().encode(
    x=alt.X('metric', axis=alt.Axis(labelAngle=0), title=''),
    y=alt.Y('value', title='')
)
st.altair_chart(bar, use_container_width=True)

with st.sidebar:

    st.markdown('''<style>
        body.nav-link {
          background: black;
          font-family: 'Source Sans Pro', sans-serif;
          font-size: 18px;
          font-weight: 400;
          text-transform: uppercase;
          letter-spacing: .1em;
        }
    
        a.nav-link {
          display: block;
          color: #2E2F3B;
          text-decoration: none;
          position: relative;
          padding: 5px 10px;
          overflow: hidden;
        }
    
        a.nav-link::after {
          content: "";
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: rgba(200, 200, 200, 0.3); /* Gris semitransparente */
          z-index: -1;
          transform: scaleX(0);
          border-radius: 8px;
        }
    
        a.nav-link:hover::after {
          transform: scaleX(1);
        }
    </style>
    ''', unsafe_allow_html=True)
    st.markdown("<a href='#model-structure' class='nav-link'>Model structure</a>", unsafe_allow_html=True)
    st.markdown("<a href='#training-history' class='nav-link'>Training history</a>", unsafe_allow_html=True)
    st.markdown("<a href='#test-values' class='nav-link'>Test values</a>", unsafe_allow_html=True)