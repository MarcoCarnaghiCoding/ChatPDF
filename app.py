import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""

    for pdf_doc in pdf_docs:
        # init pdf reader object
        pdf_reader = PdfReader(pdf_doc)
        # Loop through pages
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(raw_text, chunk_size=1000):
    #initialize splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-large')   
    vector_store = FAISS.from_texts(chunks,
                                    embeddings = embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-xxl",
        model_kwargs = {
            "temperature":0.5,
            "max_length": 256
        }
    )
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_message= True)
    conversation_chain = ConversartionalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    
    return

def main():
    # Load environment variables
    load_dotenv()

    #set page configuration
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
    )

    st.write(css,unsafe_allow_html=True)

    #Initialize the conversation state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")      
    user_question = st.text_input("Ask a question about your PDFs:")
    
    #Salutation
    st.write(user_template.replace("{{MSG}}","Hi, ChatPDF!"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hi, there!"),unsafe_allow_html=True)

    #Chatting
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs",
                                    accept_multiple_files=True)
        
        if st.button('Process'):
            # generate spinner
            with st.spinner("Processing your PDFs ..."):
                #1. get pdf raw text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text) #debugging    
                
                #2. get pdf chuncks
                text_chunks = get_text_chunks(raw_text,
                                              chunk_size=1000)

                #st.write(text_chunks) #debugging  
                
                
                #3. create vector store /embeddings
                vector_store = get_vector_store(text_chunks)

                #4. Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()