import os
import streamlit as st
import PyPDF2
import docx
import pandas as pd
from io import BytesIO
from groq import Groq  # Assuming you're using Groq for API requests

# Get API key from Streamlit secrets
api_key = st.secrets["api_key"]

# Function to extract text from PDF and split into chunks based on token limit
def extract_text_from_pdf_in_chunks(pdf_file, token_limit=8000):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text into chunks based on token_limit
    text_chunks = []
    for i in range(0, len(text), token_limit):
        text_chunks.append(text[i:i+token_limit])
    
    return text_chunks

# Function to extract text from Word file and split into chunks
def extract_text_from_word_in_chunks(doc_file, token_limit=8000):
    doc = docx.Document(doc_file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    # Split text into chunks based on token_limit
    text_chunks = []
    for i in range(0, len(text), token_limit):
        text_chunks.append(text[i:i+token_limit])

    return text_chunks

# Function to extract text from Excel file and split into chunks
def extract_text_from_excel_in_chunks(excel_file, token_limit=8000):
    df = pd.read_excel(excel_file)
    text = df.to_string()

    # Split text into chunks based on token_limit
    text_chunks = []
    for i in range(0, len(text), token_limit):
        text_chunks.append(text[i:i+token_limit])

    return text_chunks

# Streamlit UI
st.title("RAG Based Document Question Answering and Summarization AI Chatbot")

# File Upload (PDF, Word, Excel)
uploaded_file = st.file_uploader("Upload a document (PDF, Word, Excel)", type=["pdf", "docx", "xlsx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        with st.spinner("Extracting text from the PDF..."):
            chunks = extract_text_from_pdf_in_chunks(BytesIO(uploaded_file.read()))
    elif file_type == 'docx':
        with st.spinner("Extracting text from the Word document..."):
            chunks = extract_text_from_word_in_chunks(BytesIO(uploaded_file.read()))
    elif file_type == 'xlsx':
        with st.spinner("Extracting text from the Excel file..."):
            chunks = extract_text_from_excel_in_chunks(BytesIO(uploaded_file.read()))
    else:
        st.error("Unsupported file format")

    # Option to summarize the document
    if st.button("Summarize Document"):
        if api_key is None:
            st.error("The API key is not set")
        else:
            # Initialize the Groq client
            client = Groq(api_key=api_key)

            try:
                summary_content = ""

                # Iterate over chunks and request a summary for each
                for chunk in chunks:
                    # Create a prompt to summarize the current chunk
                    context = f"Here is a portion of the document: {chunk}\n\nPlease provide a short summary of this text."

                    # Make a request to the chat completions endpoint with the document chunk
                    with st.spinner("Generating summary..."):
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an assistant that summarizes text from documents.",
                                },
                                {
                                    "role": "user",
                                    "content": context,
                                }
                            ],
                            model="llama3-8b-8192",
                        )

                        # Append the summary from the current chunk
                        summary_content += chat_completion.choices[0].message.content + "\n"

                st.success("Summary generated successfully!")
                st.write(summary_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Prompt the user for a query
    user_query = st.text_input("Enter your question about the content of the document")

    if user_query and st.button("Get Answer"):
        if api_key is None:
            st.error("The API key is not set")
        else:
            # Initialize the Groq client
            client = Groq(api_key=api_key)

            try:
                response_content = ""
                previous_context = ""

                # Iterate over chunks and make a request for each one
                for chunk in chunks:
                    # Combine the previous context and the current chunk and the user's query
                    context = f"Here is the content of the document: {previous_context + chunk}\n\nUser's question: {user_query}"

                    # Make a request to the chat completions endpoint with the document context and user input
                    with st.spinner("Generating response..."):
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an assistant that answers questions based on document content.",
                                },
                                {
                                    "role": "user",
                                    "content": context,
                                }
                            ],
                            model="llama3-8b-8192",
                        )

                        # Append the response from the current chunk
                        response_content += chat_completion.choices[0].message.content + "\n"

                        # Update previous_context by adding the current chunk, up to the token limit
                        previous_context += chunk
                        if len(previous_context) > 8000:
                            previous_context = previous_context[-8000:]  # Keep the last 8000 characters
                
                st.success("Response generated successfully!")
                st.write("Response:", response_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")
