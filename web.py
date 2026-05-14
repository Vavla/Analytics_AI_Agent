import streamlit as st
import pandas as pd
import numpy as np

st.title('Analytics AI Agent')

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader(
    "Upload data", type="csv", key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    if st.button("Process file"):
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write(st.session_state.df)
        except pd.errors.ParserError:
            st.write("Ошибка чтение файла. Проверьте, что данные в файле верные, или загрузите другой файл")

if st.button("Clear and upload new file"):
    st.session_state.df = None
    st.session_state.uploader_key += 1
    st.rerun()

prompt_enter = st.text_input("Write your personal prompt: ")
st.write("You want the system to take into account the following request: ")
st.write(prompt_enter)


