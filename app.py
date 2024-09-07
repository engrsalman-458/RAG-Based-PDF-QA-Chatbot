
# Get API key from Streamlit secrets
api_key = st.secrets["api_key"]

# Function to extract text from PDF and limit token usage
def extract_text_from_pdf(pdf_file, limit=10000):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
        if len(text) > limit:  # Limit the length of extracted text
            break
    return text

# Streamlit UI
st.title("RAG Based PDF Question Answering")

# PDF File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))

    st.success("Text extracted successfully!")

    # Display the extracted text (optional)
    if st.checkbox("Show extracted PDF text"):
        st.write(pdf_text)

    # Prompt the user for a query
    user_query = st.text_input("Enter your question about the content of the PDF")

    if user_query and st.button("Get Answer"):
        if api_key is None:
            st.error("The API key is not set")
        else:
            # Initialize the Groq client
            client = Groq(api_key=api_key)

            # Combine the PDF text and the user's query
            context = f"Here is the content of the PDF: {pdf_text}\n\nUser's question: {user_query}"

            # Make a request to the chat completions endpoint with the PDF context and user input
            try:
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

                    # Display the response
                    response_content = chat_completion.choices[0].message.content
                    st.success("Response generated successfully!")
                    st.write("Response:", response_content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
