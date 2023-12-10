import streamlit as st
import os
from PIL import Image


@st.cache_data
def load_image(photo):
    return Image.open(photo)


st.header('Image Color Segmentation')
st.write('A simple app to see top 5 most dominating colors in your image')

image_file = st.file_uploader(label='upload an image', type=['jpeg', 'png', 'jpg'])

if image_file is not None:
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    img = load_image(image_file)
    st.image(img, width=250)

    with open(os.path.join("data/raw/", image_file.name), "wb") as f:
        f.write(image_file.getbuffer())
    st.success('Successfully Uploaded !')