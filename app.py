import streamlit as st

st.header('Image Color Segmentation')
st.write('A simple app to see top 5 most dominating colors in your image')

file = st.file_uploader(label='upload an image')