import streamlit as st

def main_sidebar():
    st.sidebar.success("Select a demo above.")

    st.sidebar.markdown(
        """
        - [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction)
        - [LangChain Installation](https://python.langchain.com/docs/get_started/installation)
        - [LangChain Security](https://python.langchain.com/docs/security)
        - [LangChain Version 1.0](https://python.langchain.com/v0.1/docs/get_started/introduction/)
        - [Streamlit iframe API](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.iframe)
    """
    )

def LC_QuickStart_00_sidebar():
    st.sidebar.header("AI Web App Development 🏠")
    st.sidebar.markdown(
        """
        Tool : LangChain, Streamlit, OpenAI, Python, Visual Studio Code, Streamlit Cloud, Github
        \nThis page is all about sharing what I have found while digging into the tools and methods I need to develop my AI Web App.    
    """
    )
    st.sidebar.markdown(
        """
        ## Links where you can download the necessary tools.

        - [Python](https://www.python.org/downloads/)
        - [OpenAI API Key](https://platform.openai.com/api-keys)
        - [Visual Studio Code](https://code.visualstudio.com/download)
        - [LangChain](https://python.langchain.com/docs/get_started/installation)
        - [Streamlit](https://docs.streamlit.io/get-started/installation)
        - [Streamlit Cloud](https://streamlit.io/cloud)
        - [Streamlit Multipage Template](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app)
        - [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
        - [HuggingFace](https://huggingface.co/)
        - [Kaggle](https://www.kaggle.com/)
        - [Github Docs](https://docs.github.com/en/get-started/start-your-journey/hello-world)
    """
    )

def LC_QuickStart_01_sidebar():
    st.sidebar.header("LangChain QuickStart 01 🧑‍🎨")
    st.sidebar.markdown(
        """
        Tool : ChatOpenAI, Langchain, Streamlit, ChatPromptTemplate, StrOutputParser, openai-OpenAIError
        \nOn this page, you will learn how to build a simple application with LangChain and how to use the most basic and common components of LangChain: prompt templates, models, and output parsers.
    """
    )
    st.sidebar.markdown(
        """
        ## Items to study in this example:

        - [LangChain QuickStart](https://python.langchain.com/v0.1/docs/get_started/quickstart)
        - [ChatOpenAI Python](https://python.langchain.com/docs/integrations/chat/openai)
        - [ChatOpenAI JS](https://js.langchain.com/docs/integrations/chat/openai)
        - [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
        - [StrOutputParser API](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
        - [OpenAI Error Types](https://help.openai.com/en/articles/6897213-openai-library-error-types-guidance)
    """
    )
