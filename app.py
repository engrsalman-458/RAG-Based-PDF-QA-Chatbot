import os
import streamlit as st
import PyPDF2
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

# Function to summarize the PDF chunks within a word limit
def summarize_pdf_chunks(client, pdf_chunks):
    summary_content = ""

    for chunk in pdf_chunks:
        # Modify the context to ask for a summary with a word limit
        context = f"Here is a portion of the PDF: {chunk}\n\nPlease provide a concise summary between 50 to 100 words."

        # Make a request to the chat completions endpoint
        with st.spinner("Generating summary..."):
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that summarizes text from a PDF in a concise manner, with summaries limited to 50-100 words.",
                    },
                    {
                        "role": "user",
                        "content": context,
                    }
                ],
                model="llama3-8b-8192",
                max_tokens=200  # You can adjust this if needed
            )

            # Retrieve the summary and ensure it fits within 50-100 words
            summary = chat_completion.choices[0].message.content.strip()
            words = summary.split()

            if len(words) > 100:
                summary = " ".join(words[:100]) + "..."
            elif len(words) < 50:
                summary += " (The summary is below 50 words, please provide a longer summary for completeness.)"
            
            # Append the processed summary
            summary_content += summary + "\n"
    
    return summary_content

# Streamlit UI
st.title("RAG Based PDF Question Answering and Summarization AI Chatbot")

# PDF File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
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
                # Summarize the PDF chunks with a 50-100 word limit
                summary_content = summarize_pdf_chunks(client, pdf_chunks)
                
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
