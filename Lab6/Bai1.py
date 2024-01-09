import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


def Bai1():
    # Upload file dữ liệu
    st.title("Bài 1")
    data_file = st.file_uploader("Upload file dữ liệu", type=["csv"])
    if data_file is not None:
        data = load_data(data_file)
        st.write(data)
        # Thống kê mô tả dữ liệu
        st.header("Thống kê mô tả dữ liệu")
        st.write(data.describe())
        # Mô tả các thuộc tính
        st.header("Mô tả các thuộc tính")
        st.write(data.info())
        # Vẽ biểu đồ histogram biểu diễn sự phân bố giá trị của các thuộc tính
        st.header("Vẽ biểu đồ histogram biểu diễn sự phân bố giá trị của các thuộc tính")
        for col in data.columns:
            if data[col].dtype != object:
                st.write(col)
                plt.hist(data[col])
                st.pyplot()
        # Tính hệ số tương quan giữa các thuộc tính
        st.header("Tính hệ số tương quan giữa các thuộc tính")
        st.write(data.corr())
        # Chọn biến phụ thuộc và vẽ biểu đồ phân tán biểu diễn mối liên hệ giữa biến phụ thuộc và từng biến độc lập
        st.header("Chọn biến phụ thuộc và vẽ biểu đồ phân tán biểu diễn mối liên hệ giữa biến phụ thuộc và từng biến độc lập")
        col = st.selectbox("Chọn biến phụ thuộc", data.columns)
        for col1 in data.columns:
            if data[col1].dtype != object and col1 != col:
                st.write(col1)
                plt.scatter(data[col1], data[col])
                st.pyplot()

                