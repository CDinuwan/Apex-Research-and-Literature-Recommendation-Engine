def run_app():
    import streamlit as st
    from openai import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import ConversationChain, LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    import json
    import time
    import tempfile

    # Set page config
    st.set_page_config(page_title="Wiley ScholarScope", layout="wide")

    # Custom CSS for styling
    st.markdown("""
    <style>
        body {
            color: #333;
            background-color: #f0f8ff;
        }
        .stApp {
            background-image: linear-gradient(to bottom right, #e6f2ff, #ffffff);
        }
        .stButton>button {
            color: #ffffff;
            background-color: #4682b4;
            border-radius: 5px;
        }
        .stMultiSelect>div>div>div {
            background-color: #f0f8ff;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for API configuration
    with st.sidebar:
        st.header("API Configuration")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")

        # File uploader for research papers
        st.header("Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )

    # Initialize clients and models
    if api_key:
        client = OpenAI(api_key=api_key)
        llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    else:
        client = None
        llm = None
        embeddings = None

    # Process uploaded files
    def process_uploaded_files(files, embeddings):
        documents = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name

                if file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents.extend(loader.load())

        if documents:
            texts = text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, embeddings)
            return vector_store
        return None

    # Process uploaded files if any
    if uploaded_files and embeddings:
        with st.spinner("Processing uploaded documents..."):
            st.session_state.vector_store = process_uploaded_files(uploaded_files, embeddings)
            st.success("Documents processed successfully!")

    # Title and description
    st.title("Wiley ScholarScope - AI-Driven Research Recommendation Engine")
    st.subheader("Cross-Domain Knowledge Enhancement with LangChain Integration")

    # User information
    user_name = st.text_input("Enter your name:")
    st.write(f"Welcome, {user_name}!")

    # Research domains
    domains = [
        "Computer Science", "Biology", "Physics", "Chemistry", "Mathematics",
        "Psychology", "Economics", "Environmental Science", "Medicine", "Engineering"
    ]

    # Multiple selection for research interests
    interests = st.multiselect(
        "Select your research interests:",
        domains,
        default=["Computer Science", "Mathematics"]
    )

    # Experience level
    experience = st.select_slider(
        "Select your experience level:",
        options=["Beginner", "Intermediate", "Advanced", "Expert"]
    )

    # Preferred content types
    content_types = st.multiselect(
        "Select preferred content types:",
        ["Research Papers", "Books", "Journals", "Conference Proceedings", "Tutorials", "Video Lectures"],
        default=["Research Papers", "Journals"]
    )

    # Research focus areas
    focus_areas = st.text_area(
        "Describe your specific research focus areas or topics of interest:",
        help="Example: Machine learning applications in healthcare, particularly in diagnostic imaging"
    )

    # Time frame
    time_frame = st.select_slider(
        "Select preferred publication time frame:",
        options=["Last 6 months", "Last year", "Last 2 years", "Last 5 years", "All time"]
    )

    # Replace the schema definition section in your code with this:

    # Define output schemas for structured parsing
    response_schemas = [
        ResponseSchema(
            name="recommendations",
            description="List of research paper recommendations",
            type="array"
        ),
        ResponseSchema(
            name="emerging_directions",
            description="List of emerging research directions",
            type="array"
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    def wait_for_run_completion(client, thread_id, run_id):
        """Wait for the assistant's run to complete"""
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                return run
            elif run.status == "failed":
                raise Exception("Run failed")
            time.sleep(1)

    def get_relevant_documents(query, vector_store):
        """Retrieve relevant documents from the vector store"""
        if vector_store:
            return vector_store.similarity_search(query, k=3)
        return []

    def generate_enhanced_recommendations(client, llm, user_data, vector_store):
        """Generate recommendations using both Assistant API and LangChain"""
        try:
            # Get relevant documents from uploaded papers
            relevant_docs = get_relevant_documents(user_data['focus_areas'], vector_store)

            # Create a thread
            thread = client.beta.threads.create()

            # Prepare the user profile message with relevant document context
            context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
            user_profile = f"""
            Please provide past researches in vector database based on this profile:

            Background:
            - Name: {user_data['name']}
            - Experience Level: {user_data['experience']}
            - Research Interests: {', '.join(user_data['interests'])}
            - Content Types of Interest: {', '.join(user_data['content_types'])}
            - Specific Focus Areas: {user_data['focus_areas']}
            - Preferred Time Frame: {user_data['time_frame']}

            Relevant Context from Uploaded Papers:
            {context}

            Provide 5-7 recommendations that cross multiple domains and 2-3 emerging research directions.
            """

            # Create LangChain conversation chain
            conversation = ConversationChain(
                llm=llm,
                memory=st.session_state.memory,
                verbose=True
            )

            # Add the message to the thread
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_profile
            )

            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id="asst_x0u72RaeQZPWRm2kd5dSlIm2"
            )

            # Wait for completion
            run = wait_for_run_completion(client, thread.id, run.id)

            # Get the assistant's response
            messages = client.beta.threads.messages.list(thread_id=thread.id)

            # Process the response with LangChain
            assistant_response = None
            for msg in messages:
                if msg.role == "assistant":
                    assistant_response = msg.content[0].text.value
                    break

            if assistant_response:
                # Enhance the response using LangChain
                enhanced_response = conversation.predict(input=f"""
                    Please enhance this recommendation with additional insights:
                    {assistant_response}
                """)
                return enhanced_response

            return assistant_response

        except Exception as e:
            raise Exception(f"Error in getting recommendations: {str(e)}")

    # Generate recommendations
    if st.button("Generate Recommendations"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return

        st.write("Generating personalized recommendations...")

        # Prepare user data
        user_data = {
            "name": user_name,
            "experience": experience,
            "interests": interests,
            "content_types": content_types,
            "focus_areas": focus_areas,
            "time_frame": time_frame
        }

        try:
            # Get enhanced recommendations
            response = generate_enhanced_recommendations(
                client,
                llm,
                user_data,
                st.session_state.vector_store
            )

            # Try to parse the response as JSON
            try:
                recommendations = json.loads(response)

                # Display recommended papers
                st.subheader("Recommended Papers:")
                if "recommendations" in recommendations:
                    for paper in recommendations["recommendations"]:
                        with st.expander(f"📄 {paper['title']}"):
                            st.write(f"**Authors:** {paper['authors']}")
                            st.write(f"**Year:** {paper['year']}")
                            st.write(f"**Relevance:** {paper['relevance']}")
                            st.write(f"**Key Findings:** {paper['key_findings']}")
                            st.write(f"**Connections:** {paper['connections']}")

                # Display research directions
                st.subheader("Emerging Research Directions:")
                if "emerging_directions" in recommendations:
                    for direction in recommendations["emerging_directions"]:
                        st.write(f"🔍 **{direction['title']}**")
                        st.write(direction['description'])

                # Display relevant documents from uploads
                if st.session_state.vector_store:
                    st.subheader("Relevant Documents from Your Uploads:")
                    relevant_docs = get_relevant_documents(focus_areas, st.session_state.vector_store)
                    for i, doc in enumerate(relevant_docs, 1):
                        with st.expander(f"📚 Related Document {i}"):
                            st.write(doc.page_content)

                st.success("Recommendations generated successfully!")

            except json.JSONDecodeError:
                # If JSON parsing fails, display the raw response
                st.write("Enhanced Recommendations:")
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Chat interface for follow-up questions
    st.write("---")
    st.subheader("Ask Follow-up Questions")
    user_question = st.text_input("Ask a question about the recommendations:")

    if user_question and llm:
        conversation = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            verbose=True
        )
        response = conversation.predict(input=user_question)
        st.write("Response:", response)

    # Feedback
    st.write("---")
    feedback = st.text_area("Provide feedback on the recommendations:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

    # Footer
    st.write("---")
    st.write("© 2024 AI-Driven Research Recommendation Engine")


if __name__ == "__main__":
    run_app()