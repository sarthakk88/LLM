import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# find or create the embeddings
def get_embeddings(text_chunks=" "):
    # check if vector store already exists
    # if REUSE_PKL_STORE is True, then load the vector store from disk if it exists
    reuse_pkl_store = os.getenv("REUSE_PKL_STORE")
    if reuse_pkl_store == "True" and os.path.exists(DB_FAISS_PATH):
        with open(DB_FAISS_PATH, "rb") as f:
            vector_store = pickle.load(f)
        st.write("Embeddings loaded from disk")
    #else create embeddings and save to disk
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        # save the vector store to disk
        with open(DB_FAISS_PATH, "wb") as f:
            pickle.dump(vector_store, f)
        st.write("Embeddings saved to disk")
    if vector_store is not None:
        return vector_store
    else:
        raise ValueError("Issue creating and saving vector store")

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     # vector_store.save_local(DB_FAISS_PATH)
#     with open(DB_FAISS_PATH, "wb") as f:
#         pickle.dump(vector_store, f)


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    vector_store = get_embeddings()
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]




def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        if 'GOOGLE_API_KEY' in st.secrets:
            st.success('API key already provided!', icon='✅')
            GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
        else:
            GOOGLE_API_KEY = st.text_input('Enter Gemini API Key:', type='password')
            if not GOOGLE_API_KEY:
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()