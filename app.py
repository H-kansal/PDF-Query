import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_TOKEN']=st.secrets["HF_TOKEN"]
st.title("PDF Q&A ChatBot with Groq and Ollama")
st.write("Upload Pdf's and chat with their content")

api_key=st.text_input("Your Groq Api Key",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model="openai/gpt-oss-20b")
    session_id=st.text_input("sessionId",value="default session")
    if "store" not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Upload your Pdfs",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        document=[]
        for uploaded_file in uploaded_files:
            fileName = f"/tmp/{uploaded_file.name}"
            with open(fileName,"wb") as file:
                file.write(uploaded_file.getvalue())
            
            docs_loader=PyPDFLoader(fileName)
            docs=docs_loader.load()
            document.extend(docs)
    
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        final_docs=text_splitter.split_documents(document)
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb=Chroma.from_documents(final_docs,embedding)
        retriever=vectordb.as_retriever()
    
        model_prompt=ChatPromptTemplate.from_messages(
            [
                ("system","Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )
        
        history_retriever=create_history_aware_retriever(llm,retriever,model_prompt)

        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        document_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_retriever,document_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store["session_id"]=ChatMessageHistory()
            return st.session_state.store["session_id"]
        
        with_message_history=RunnableWithMessageHistory(rag_chain,
                get_session_history,
                input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer")
        
        user_input=st.text_input("Ask question")

        if user_input:
            response=with_message_history.invoke({
                "input":user_input
            },
            config={
                "configurable": {"session_id":session_id}
            }
            )

            st.write(response['answer'])
else:
    st.warning("Please enter api key")     
