import os
import tempfile
import streamlit as st

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persistent_directory = 'db'

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file: # Create a temporary file to store the uploaded PDF
        temp_file.write(file.read())
        temp_file_path = temp_file.name # Get the path of the temporary file

    loader = PyPDFLoader(temp_file_path) # Load the PDF file using PyPDFLoader
    docs = loader.load() # Load the documents from the PDF file

    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    ) # Split the text into chunks
    chuncks = text_splitter.split_documents(docs) # Split the loaded documents into chunks
    return chuncks # Return the list of chunks extracted from the PDF file

def load_existing_vectorstore(): # Check if a vectorstore already exists
    if os.path.exists(os.path.join(persistent_directory)):
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vectorstore # Return the existing vectorstore if it exists
    return None # Create a new vectorstore if it does not exist

def add_to_vectorstore(chuncks, vectorstore=None):
    if vectorstore:
        vectorstore.add_documents(chuncks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chuncks, 
            embedding=OpenAIEmbeddings(),
            persist_directory=persistent_directory
        )
    return vectorstore # Add the chunks to the vectorstore and return it

def ask_question(model, query, vectorstore):
    llm = ChatOpenAI(model=model) # Initialize the language model
    retriever = vectorstore.as_retriever() # Create a retriever from the vectorstore / buscador de dados

    system_prompt = """ Use the context for answer the questions.
    if you dont find the answer, explain what there is no enough information avaliable.
    answer in the format of markdown and with interactive visualizations.
    answer in portuguese.
    context: {context}
    """    
    messages = [('system', system_prompt)]  # first message is the system prompt
    for message in st.session_state['messages']:
        messages.append((message.get('role'), message.get('content'))) # historic messages are added to the message list
    messages.append(('human','{input}')) # last message is the user input

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain( # create_stuff_documents_chain creates a chain that combines the documents and the LLM to answer questions
        llm=llm,
        prompt=prompt,
    )

    chain = create_retrieval_chain( 
        retriever=retriever,
        combine_docs_chain=question_answer_chain, 
    )
    response = chain.invoke({'input': query}) # Invoke the chain with the user query
    return response.get('answer') # Return the answer from the response


vectorstore = load_existing_vectorstore()

st.set_page_config(
    page_title="Helpful Assistant",
    page_icon="🤖",
)

st.header("🤖Bem vindo ao seu assistente com RAG!")

with st.sidebar:
    st.header("Upload Your File 📄")
    uploaded_file = st.file_uploader(
        label="Do upload a PDF file", 
        type=["pdf"],
        accept_multiple_files=True, # Allows multiple file uploads
        )
    
    if uploaded_file: # If a file is uploaded, process it
        with st.spinner('Processing your file...'):
            all_chuncks = []
            for uploaded_file in uploaded_file: # Loop through each uploaded file
                chuncks = process_pdf(file=uploaded_file) # Process the PDF file to extract text
                all_chuncks.extend(chuncks) # get all chunks from the processed files and add them to a list chuncks
            vectorstore = add_to_vectorstore(
                chuncks=all_chuncks, 
                vectorstore=vectorstore,
            )
    
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    selected_models = st.sidebar.selectbox(
        label='Selecione o modelo de LLM',
        options=model_options,
    )

if 'messages' not in st.session_state: # if no messages in session state, initialize it
    st.session_state['messages'] = [] # save the messages in the session state

question = st.chat_input('como eu posso te ajudar?')

if vectorstore and question:
    for message in st.session_state['messages']:
            st.chat_message(message.get('role')).write(message.get('content')) # remonta o historico de mensagem

    st.chat_message('user').write(question) # add the user question to the chat message
    st.session_state['messages'].append({'role': 'user', 'content': question}) # append the user question to the session state
        
    with st.spinner('Buscando a melhor resposta...'):
        response = ask_question(
            model=selected_models,
            query=question,
            vectorstore=vectorstore,
        ) # Call the function to ask a question and get a response

        st.chat_message('ai').write(response)
        st.session_state['messages'].append({'role': 'ai', 'content': response}) # Append the response to the session state
        







