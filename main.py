import streamlit as st
import streamlit.components.v1 as components
from langchain_sidebar_content import main_sidebar

st.set_page_config(
    page_title="Catchup LangChain Tutorial",
    page_icon="👋",
)

st.write("# Catchup LangChain Tutorial!👋")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    For bigger screen, click the link below.

    - [LangChain Doc](https://python.langchain.com/docs/introduction/)

"""
)

components.iframe("https://python.langchain.com/docs/introduction/", width=1024, height=1200)

st.markdown(
    """
    ## CatchUp AI related materials

    - [Catchup AI Streamlit Web App](https://catchupai.streamlit.app/)
    - [Catchup AI for Small Business App](https://catchupai4sb.streamlit.app/)
    - [Catchup AI Youtube Channel](https://www.youtube.com/@catchupai)
    - [Catchup AI Tistory Blog](https://coronasdk.tistory.com/)
    - [Deep Learning Fundamental PPT (Eng. Ver.)](https://docs.google.com/presentation/d/1F4qxSAv9g13de99rS8fcp4e1LCfrILq8QaahXCPx1Pw/edit?usp=sharing)
    - [Deep Learning Fundamental PPT (Kor. Ver.)](https://docs.google.com/presentation/d/15KNzGnSnJx_4ToSBM2MrHiC2q5MiVe0plOs7f3NJuWM/edit?usp=sharing)
    - [AI Web App Development 101 PPT](https://docs.google.com/presentation/d/18_6z05tmR_loTPWFHj8PCY3-uCNKuoy-IvE0g5ms8YM/edit?usp=sharing)
"""
)

main_sidebar()
