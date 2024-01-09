import streamlit as st
from Bai1 import Bai1
from Bai6 import Bai6
from Bai10 import Bai10

def main():
    st.title("Lab6")
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Chọn bài", ["Bài 1", "Bài 6", "Bài 10"])
    if app_mode == "Bài 1":
        Bai1()
    elif app_mode == "Bài 6":
        Bai6()
    elif app_mode == "Bài 10":
        Bai10()