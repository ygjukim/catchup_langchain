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

def LC_QuickStart_02_sidebar():
    st.sidebar.header("LangChain QuickStart 02 🧐")
    st.sidebar.markdown(
        """
        Tools : beautifulsoup4, WebBaseLoader, OpenAIEmbeddings, FAISS, RecursiveCharacterTextSplitter, create_stuff_documents_chain, create_retrieval_chain
        \nRetrieval is useful when you have too much data to pass to the LLM directly. You can then use a retriever to fetch only the most relevant pieces and pass those in.
        \nIn this process, we will look up relevant documents from a Retriever and then pass them into the prompt. A Retriever can be backed by anything - a SQL table, the internet, etc - but in this instance we will populate a vector store and use that as a retriever
    """
    )
    st.sidebar.markdown(
        """
        ## Items to study in this example:

        - [LangChain QuickStart](https://python.langchain.com/v0.1/docs/get_started/quickstart)
        - [Vector stores](https://python.langchain.com/docs/modules/data_connection/vectorstores)
        - [Bueatifulsoup](https://beautiful-soup-4.readthedocs.io/en/latest/)
        - [WebBaseLoader](https://python.langchain.com/docs/integrations/document_loaders/web_base)
        - [WebBaseLoader API](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
        - [OpenAI Embeddings](https://python.langchain.com/docs/integrations/text_embedding/openai)
        - [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
        - [FAISS API](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
        - [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
        - [RecursiveCharacterTextSplitter API](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
        - [Chains](https://python.langchain.com/docs/modules/chains)
        - [create_stuff_documents_chain API](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html)
        - [create_retrival_chain API](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html#)
    """
    )

def LC_QuickStart_02_RemedialClass_sidebar():
    st.sidebar.header("LangChain QuickStart 02 Remedial Class ✍🏻")
    st.sidebar.markdown(
        """
        Tools : RecursiveCharacterTextSplitter() params, as_retriever() prams, VectorStore
        \nIn this page, we will explore the parameters available for use in RecursiveCharacterTextSplitter() and as_retriever().
    """
    )
    st.sidebar.markdown(
        """
        ## Items to study in this example:

        - [Recursively split by character](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
        - [Blog for RecursivelyCharacterTextSplitter](https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846)
        - [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
        - [RecursiveCharacterTextSplitter API](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
        - [Vector stores](https://python.langchain.com/docs/modules/data_connection/vectorstores)
        - [Vector Store API](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html)
        - [Vector store backed retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/)
    """
    )

def LC_QuickStart_03_CoversationRetrievalChain_sidebar():
    st.sidebar.header("LangChain QuickStart 03 🗣️")
    st.sidebar.markdown(
        """
        Tools : create_history_aware_retriever, MessagesPlaceholder, HumanMessage, AIMessage
        \nThe previous chain can only handle single questions, but to accommodate follow-up questions in applications like chat bots, modifications are needed.
        \nTwo adjustments are crucial
        \n1. The retrieval method must consider the entire history, not just the latest input.
        \n2. The final LLM chain should also incorporate the entire history.
    """
    )
    st.sidebar.markdown(
        """
        ## Items to study in this example:

        - [LangChain QuickStart](https://python.langchain.com/v0.1/docs/get_started/quickstart)
        - [create_history_aware_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html)
        - [Conversational RAG](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/)
        - [How to add chat history](https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/)
        - [How to stream results from your RAG application](https://python.langchain.com/v0.2/docs/how_to/qa_streaming/)
        - [Add message history(memory)](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)
        - [ConversationalRetrievalChain API](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)
        - [Types of MessagepromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/message_prompts)
        - [MessagePlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)
        - [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html)
        - [AIMessage](https://api.python.langchain.com/en/v0.0.339/schema/langchain.schema.messages.AIMessage.html)
    """
    )
    st.sidebar.markdown(
        """
        ## Youtube Clip

        - [Youtube Clip](https://youtu.be/EQ6c9309T8c?si=pMUgmF4xMClk3SaI)
    """
    )

