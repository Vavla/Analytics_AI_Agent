import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from ai_agent import run_agent, temp_file_context

load_dotenv()


st.title('Analytics AI Agent')


if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'response' not in st.session_state:
    st.session_state.response = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'prompt_counter' not in st.session_state:
    st.session_state.prompt_counter = 0

uploaded_file = st.file_uploader(
    "Загрузка файла", 
    type="csv", 
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    if st.button("Показ контента"):
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            st.success(f"Файл загружен! {st.session_state.df.shape[0]} строк, {st.session_state.df.shape[1]} колонок")
            st.write(st.session_state.df.head())
        except pd.errors.ParserError:
            st.error("Ошибка чтения файла")

if st.button("Отчистить и загрузить заново"):
    st.session_state.df = None
    st.session_state.response = None
    st.session_state.file_name = None
    st.session_state.uploader_key += 1
    st.session_state.prompt_counter += 1
    st.rerun()

prompt_key = f"prompt_input_{st.session_state.prompt_counter}"
prompt_enter = st.text_input("Напишите свой персональный промпт: ", key=prompt_key)

if prompt_enter:
    st.info(f"Ваш промпт: {prompt_enter}")

if st.button("Новый промпт"):
    st.session_state.prompt_counter += 1
    st.session_state.response = None
    st.rerun()

st.divider()
st.subheader("Вопрос для AI Agent")


if st.button("Выполнить", type="primary") and not st.session_state.processing:
    if st.session_state.df is None:
        st.warning("Сначала загрузите файл с датасетом")
    else:
        st.session_state.processing = True
        st.rerun()

if st.session_state.processing:
    with st.spinner("AI agent обрабатывает данные..."):
        try:
            with temp_file_context(uploaded_file) as temp_path:
                response = run_agent(
                    file_path=temp_path,
                    file_name=st.session_state.file_name,
                    user_instruction=prompt_enter if prompt_enter else ""
                )
            if response is False:
                st.session_state.response = "Запрос отклонён: обнаружена промпт-инъекция"
            else:
                st.session_state.response = response
            st.session_state.processing = False
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.session_state.response = f"Ошибка: {str(e)}"
            st.session_state.processing = False
            st.rerun()

# Отображение ответа
if st.session_state.response:
    st.subheader("Ответ от AI agent:")
    if prompt_enter == '':
        st.markdown("**Вы не ввели персональный промпт, агент выведет только основную информацию о датасете и его метриках.**")
    st.success(st.session_state.response)


