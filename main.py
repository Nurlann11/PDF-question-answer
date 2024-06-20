from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import openai
from langchain.chains.question_answering import load_qa_chain
def main():
    load_dotenv()

    # giving header and title to page
    st.set_page_config('pdf_ques_ans')
    st.header('pdf_ques_ans')

    # pdf downloading:
    pdf = st.file_uploader('Fayli yukle',type='pdf')
    # pdf reading: 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        txt = ''
        for page in pdf_reader.pages:
            txt += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator='\n',chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(txt)
    # st.write(chunks)

    # Creating Embeddings:
    embeddings = OpenAIEmbeddings()
    knowledge_base =  FAISS.from_texts(chunks,embeddings)

    # istifadeci sualinin bolumu:
    user_question = st.text_input("pdfden sualini ver:")
    if user_question: 
        docs = knowledge_base.similarity_search(user_question)

    llm = openai()
    chain = load_qa_chain(llm=llm, chain_type='stuff')
    response = chain.run(question =user_question, context= docs)
    st.write(response)

if __name__ == '__main__':
    main()