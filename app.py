import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import os


with st.sidebar:
    st.title("🦜️🔗 BITS chat PDF")
    st.markdown(""" 
    ## About
    This app is an LLM-powered chatbot build using:
    """)
    add_vertical_space(5)


def main():
    st.header("chat wiht pdf")
    load_dotenv()

    pdf =st.file_uploader('File uploader',type="pdf")
    if pdf:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        
        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                Vectorstore=pickle.load(f)
        else:
            embeddings=OpenAIEmbeddings()
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(Vectorstore,f)
        query=st.text_input("question regarding pdf file:")
        if query:
            docs=Vectorstore.similarity_search(query,k=3)
            llm=OpenAI(model_name="gpt-3.5-turbo",temperature=0)
            chain=load_qa_chain(llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)   


            






if __name__== "__main__":
    main()
