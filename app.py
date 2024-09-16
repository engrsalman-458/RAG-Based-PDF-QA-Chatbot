import os
import streamlit as st
import PyPDF2
import time  # For implementing a retry mechanism
from io import BytesIO
from groq import Groq  # Assuming you're using Groq for API requests

# Get API key from Streamlit secrets
api_key = st.secrets["api_key"]

# Function to extract text from PDF and split into smaller chunks
def extract_text_from_pdf_in_chunks(pdf_file, token_limit=3000):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text into smaller chunks based on a reduced token limit
    text_chunks = []
    for i in range(0, len(text), token_limit):
        text_chunks.append(text[i:i+token_limit])

    return text_chunks

# Function to summarize PDF chunks with retry for rate limit
def summarize_pdf_with_retry(client, pdf_chunks, word_limit=100, retry_delay=180):
    summary_content = ""

    for chunk in pdf_chunks:
        context = f"Here is a portion of the PDF: {chunk}\n\nPlease provide a concise summary between 50 to {word_limit} words."

        while True:
            try:
                # Request summarization with the Groq API
                with st.spinner("Generating summary..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are an assistant that summarizes text from a PDF in a concise manner, with summaries limited to 50-{word_limit} words.",
                            },
                            {
                                "role": "user",
                                "content": context,
                            }
                        ],
                        model="llama3-8b-8192",
                        max_tokens=200  # Adjust token limit to control output size
                    )
                
                # Append the summary
                summary_content += chat_completion.choices[0].message.content + "\n"
                break  # Exit loop on success

            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg:
                    # Extract wait time from error message (in this case, we hardcode to retry after 3 minutes)
                    st.warning(f"Rate limit reached. Retrying after {retry_delay//60} minutes...")
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    st.error(f"An error occurred: {e}")
                    return ""  # Exit on other errors

    return summary_content

# Streamlit UI
st.title("RAG Based PDF Question Answering and Summarization AI Chatbot")

# PDF File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file in chunks
    with st.spinner("Extracting text from the PDF..."):
        pdf_chunks = extract_text_from_pdf_in_chunks(BytesIO(uploaded_file.read()))

    # Option to summarize the PDF
    if st.button("Summarize PDF"):
        if api_key is None:
            st.error("The API key is not set")
        else:
            # Initialize the Groq client
            client = Groq(api_key=api_key)

            try:
                # Summarize the PDF with retry mechanism in case of rate limit
                summary_content = summarize_pdf_with_retry(client, pdf_chunks)
                
                st.success("Summary generated successfully!")
                st.write(summary_content)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Prompt the user for a query
    user_query = st.text_input("Enter your question about the content of the PDF")

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
                for chunk in pdf_chunks:
                    # Combine the previous context and the current chunk and the user's query
                    context = f"Here is the content of the PDF: {previous_context + chunk}\n\nUser's question: {user_query}"

                    # Make a request to the chat completions endpoint with the PDF context and user input
                    with st.spinner("Generating response..."):
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an assistant that answers questions based on PDF content.",
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
