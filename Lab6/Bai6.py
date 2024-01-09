import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st

model = load_model('base_model_best.h5')

class_names = ['Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh',
              'Banh chung', 'Banh cuon', 'Banh duc',
              'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 
              'Banh tet', 'Banh trang nuong', 'Banh xeo', 
              'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 
              'Bun rieu', 'Bun thit nuong', 'Ca kho to', 
              'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 
              'Goi cuon', 'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 
              'Xoi xeo']

def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def Bai6():
    st.title("Bài 6")
    img_file = st.file_uploader("Upload image", type=["jpg", "png"])
    if img_file is not None:
        img = load_img(img_file)
        result = model.predict(img)
        st.write('Predict result:')
        st.write(class_names[np.argmax(result)])
        st.image(img_file, width=300)
        