# --- 1. Setup & Imports ---
import os  # For environment variable management
import streamlit as st  # For building the web app UI
import pandas as pd  # For data manipulation (not directly used, but often useful with CSVs)
from dotenv import load_dotenv  # For loading environment variables from a .env file
from langchain_community.document_loaders import CSVLoader  # To load documents from a CSV file
from langchain_community.embeddings import OpenAIEmbeddings  # For generating embeddings using OpenAI
from langchain_community.vectorstores import Chroma  # For storing and retrieving document embeddings
from langchain_community.llms import Ollama  # For using the Ollama LLM (e.g., LLaMA2)
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain.chains.combine_documents import create_stuff_documents_chain  # For combining docs and LLM
from langchain.chains import create_retrieval_chain  # For creating a retrieval-augmented chain

# --- 2. Load .env & Set API Keys ---
load_dotenv()  # Load environment variables from .env file
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY_4")  # Set OpenAI API key for embeddings

# --- 3. Load CSV & Vector Store (reused from your code) ---
loader = CSVLoader('symptoms_data.csv')  # Load the CSV file containing symptom data
docs = loader.load()  # Parse the CSV into document objects
db = Chroma.from_documents(docs, OpenAIEmbeddings())  # Create a Chroma vector store from the documents

# --- 4. LLM + Prompt Setup ---
llm = Ollama(model="llama2")  # Initialize the Ollama LLM (LLaMA2)
prompt = ChatPromptTemplate.from_template("""
Answer the question only with a list of 3 to 5 clear follow-up questions related to the symptom mentioned. Do NOT include any explanations or introductory text.

<context>
{context}
</context>
Question: {input}
""")  # Prompt template for generating follow-up questions
document_chain = create_stuff_documents_chain(llm, prompt)  # Chain to combine docs and LLM with prompt
retriever = db.as_retriever()  # Create a retriever from the vector store
retrieval_chain = create_retrieval_chain(retriever, document_chain)  # Retrieval-augmented QA chain

# --- 5. Initialize Streamlit Session State ---
# Ensure session state variables are initialized for multi-step interaction
if 'symptom' not in st.session_state:
    st.session_state.symptom = ''
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'follow_ups' not in st.session_state:
    st.session_state.follow_ups = []
if 'step' not in st.session_state:
    st.session_state.step = 'ask_symptom'

# --- 6. Streamlit Title ---
st.title("AI Diagnostic Assistant")  # App title

# --- 7. Step 1: Ask for Main Symptom ---
if st.session_state.step == 'ask_symptom':
    # Prompt user to enter their main symptom
    symptom_input = st.text_input("Describe your symptom (e.g., 'headache'):")

    if symptom_input:
        st.session_state.symptom = symptom_input  # Store symptom
        query = f"What are the follow up questions for {symptom_input}?"  # Formulate query
        result = retrieval_chain.invoke({"input": query})  # Get follow-up questions from LLM
        
        # Parse the LLM output into a list of questions
        follow_ups = result['answer'].split('\n')
        follow_ups = [q.strip("- ").strip() for q in follow_ups if q.strip() and "?" in q]
        st.session_state.follow_ups = follow_ups  # Store follow-up questions
        st.session_state.step = 'ask_followups'  # Move to next step
        st.rerun()  # Rerun to update UI

# --- 8. Step 2: Ask Follow-Up Questions Dynamically ---
elif st.session_state.step == 'ask_followups':
    if st.session_state.follow_ups:
        current_index = len(st.session_state.answers)  # Track which follow-up to ask next
        if current_index < len(st.session_state.follow_ups):
            current_q = st.session_state.follow_ups[current_index]  # Get current follow-up question
            st.write(f"Diagnostic Agent: {current_q}")  # Display question
            user_followup = st.text_input("Your answer:", key=f"answer_{current_index}")  # Get user answer

            if user_followup:
                st.session_state.answers.append(user_followup)  # Store answer
                st.rerun()  # Ask next follow-up or move on
        else:
            st.session_state.step = 'final_response'  # All follow-ups answered
            st.rerun()
    else:
        st.write("No follow-up questions found for this symptom.")  # Handle no follow-ups case
        st.session_state.step = 'final_response'
        st.rerun()

# --- 9. Step 3: Final Recommendation + Explanation ---
elif st.session_state.step == 'final_response':
    # Summarize user input and answers
    summary = f"""The user reported: {st.session_state.symptom}.
    Follow-up answers: {st.session_state.answers}"""

    # Prepare queries for LLM: one for recommendation, one for explanation
    recommendation_query = f"Based on the symptom '{st.session_state.symptom}' and the answers {st.session_state.answers}, what is the likely diagnosis and recommendation?"
    explanation_query = (
    f"A patient reported the symptom '{st.session_state.symptom}' and gave the following answers to follow-up questions: {st.session_state.answers}. "
    "Please explain the likely diagnosis and reasoning. Do not repeat the follow-up questions or ask new ones."
)


    # Run both queries through the retrieval chain
    diagnosis = retrieval_chain.invoke({"input": recommendation_query})['answer']
    explanation = retrieval_chain.invoke({"input": explanation_query})['answer']

    st.subheader("Recommendation Agent")
    st.write(diagnosis)  # Show diagnosis and recommendation

    st.subheader("Explanation Agent")
    st.write(explanation)  # Show explanation of reasoning

    # Button to reset the app and start over
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            st.rerun() 