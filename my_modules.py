import streamlit as st
import os
from lunarcalendar import Lunar, Solar, Converter

def modelName():
    return 'gpt-3.5-turbo-0125'

def modelName_embedding_small():
    return 'text-embedding-3-small'

def display_source_code(fileName):
    # Get the current file path
    current_file_path = os.path.abspath(__file__)

    # Get the path of source file in the 'pages' folder
    pages_file_path = os.path.join(os.path.dirname(current_file_path), 'pages', fileName)

    # Open the source file and read the source code
    with open(pages_file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    # Display the source code
    st.code(source_code, language='python')

def view_source_code(fileName):
    # Create a session state variable with the ket 'show_source_code' if it doesn't exist
    if 'show_source_code' not in st.session_state:
        st.session_state.show_source_code = False

    # Create a button to toggle the source code display
    if st.button('Toggle Source Code'):
        st.session_state.show_source_code = not st.session_state.show_source_code

    # Display or hide the source code based on the session state
    if st.session_state.show_source_code:
        display_source_code(fileName)
 
def show_source_code(absFilePath):
    with st.expander("Show source code"):
        # absFilePath = os.path.abspath(__file__)
        with open(absFilePath, 'r', encoding='utf-8') as f:
            code = f.read()        
        st.code(code, language='python')

def lunar_to_solar(year, month, day, is_leap=False):
    """음력 날짜를 양력 날짜로 변환하는 함수

    Args:
        year: 음력 연도 (정수)
        month: 음력 월 (정수)
        day: 음력 일 (정수)
        is_leap: 윤달 여부 (True/False)

    Returns:
        양력 날짜 (datetime.date 객체)
    """
    try:
        lunar_date = Lunar(year, month, day, is_leap)
        solar_date = Converter.Lunar2Solar(lunar_date) # directly convert
        return solar_date
    except ValueError:
        return None  # Handle invalid lunar date