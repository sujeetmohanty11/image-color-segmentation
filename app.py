import streamlit as st
import os
from PIL import Image
import cv2
from models.main import predict_color
import time


@st.cache_data
def load_image(photo):
    return Image.open(photo)


st.header('Image Color Segmentation')
st.write('A simple app to see top 5 most dominating colors in your image')

image_file = st.file_uploader(label='upload an image', type=['jpeg', 'png', 'jpg'])

if image_file is not None:
    file_details = {"FileName": image_file.name, "FileType": image_file.type}
    img = load_image(image_file)
    st.image(img, width=300)

    # saving uploaded file with name -> colors.jpg in data/raw directory
    with open(os.path.join("data/raw/input.jpg"), "wb") as f:
        f.write(image_file.getbuffer())
    st.success('Successfully Uploaded !')

# Progress bar
bar = st.progress(0)

for i in range(1, 101):
    time.sleep(0.1)
    bar.progress(value=i)

st.subheader("Top 5 most dominant colors are: ")
pic = cv2.imread("data/raw/input.jpg")
predict_color(pic)
st.image('data/processed/output.jpg', width=500)
