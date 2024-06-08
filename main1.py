import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from collections import Counter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import openai
from datetime import datetime
import sqlite3
import time
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load and split documents
loader = DirectoryLoader('data', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = pdf_splitter.split_documents(documents)


def get_retriver(apiai_key):
    #OPENAI_API_KEY=gpt_setting()
    # Initialize embeddings and vector store
    embedding = OpenAIEmbeddings(api_key=apiai_key)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding)
    retriever = vectordb.as_retriever()
    print(retriever)
    return retriever

####################################################################################################
# Set up RetrievalQA chain


@st.experimental_dialog("API 키 입력")
def gpt_setting():
    openapi = st.text_input(label="OPEN API 키", placeholder="Enter Your API Key", value="")
    if st.button("저장"):
        st.session_state["OPEN_API"] = openapi
        print(openapi)
        st.rerun()

# def create_qa_chain():
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
#         chain_type="stuff",
#         retriever=get_retriver(st.session_state["OPEN_API"]),
#         return_source_documents=True)

def ask_gpt(prompt):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
        chain_type="stuff",
        retriever=get_retriver(st.session_state["OPEN_API"]),
        return_source_documents=True)
    print("prompt")
    print(prompt)
    response=qa_chain.invoke(prompt)
    print("response")
    print(response['result'])
    return response['result']

# re=ask_gpt("인공지능학부는 졸업을 위해 몇 학점이 필요해?")
# print(re)


def main():

    st.set_page_config(
        page_title="인공지능 개발발", layout="wide")

    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "OPEN_API" not in st.session_state:
        st.session_state["OPEN_API"] = ""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role":"system", "content":"You are a thoughtful assistant. You must answer in Korean. Remember ACCURACY is the most important thing."}]
    # if "Question" not in st.session_state:
    #     st.session_state["Question"] = False
    

    st.header("신입생 GoGoGo", divider="rainbow")

    with st.sidebar:
        
        st.subheader("", divider="gray")

        col1, col2 = st.columns(2)
        with col1:
            color_var = st.color_picker("대화 색", "#ffd700")
        with col2:
            word_color = st.radio("글자 색", options=["black", "white"])

        st.subheader("", divider="gray")

        col3, col4= st.columns(2)
        with col3:
            if st.button("API 키 입력"):
                gpt_setting()
        with col4:
            if st.button(label="초기화"):
                question=""
                st.session_state["chat"] = []
                st.session_state["messages"] = [{"role":"system", "content":"You are a thoughtful assistant. You must answer in Korean. Remember ACCURACY is the most important thing."}]
                st.session_state["check_reset"] = True
                

        st.subheader("", divider="gray")

        col5, col6 = st.columns(2)
        with col5:
            with open("./data/gogogo.pdf", "rb") as file:
                btn = st.download_button(
                    label="파일 다운로드",
                    data=file,
                    file_name="gogogo.pdf",
                    mime="gogogo/pdf")
        
        with col6:
            st.link_button("학사 안내 링크", "https://web.kangnam.ac.kr/menu/3d1345da241450c621fe02936aeb96e7.do?encMenuSeq=8390760a0e57bc14a62df749f32e32f9")
        
        st.subheader("", divider="gray")
    #########################사이드 바############################

    question = st.text_input(label="", placeholder="질문을 입력하세요", value="")
    # print(question)
    st.divider()

    # print(st.session_state["messages"])
    with st.expander("전체 채팅 보기", expanded=False):
        if question:
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"]+[("user", now, question)]
            st.session_state["messages"] = st.session_state["messages"]+[{"role":"user", "content":question}]
        if question:
            response = ask_gpt(question)
            st.session_state["messages"] = st.session_state["messages"]+[{"role":"system", "content":response}]
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"]+[("bot", now, response)]
            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    color_var = color_var
                    word_color = word_color
                    st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:{color_var};color:{word_color};border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
                else:
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
    
    

if __name__=="__main__":
    main()
