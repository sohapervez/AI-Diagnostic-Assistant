# AI Diagnostic Assistant

This project is an interactive AI-powered diagnostic assistant built with Streamlit. It leverages LLMs (such as LLaMA2 via Ollama) and retrieval-augmented generation to ask users about their symptoms, generate relevant follow-up questions, and provide a likely diagnosis and explanation based on user responses.

## Features
- **Conversational Symptom Intake:** Users describe their main symptom, and the app generates 3-5 follow-up questions to clarify the case.
- **Dynamic Q&A:** The assistant asks each follow-up question in turn, collecting user answers.
- **AI-Powered Diagnosis:** After all questions are answered, the app provides a likely diagnosis and a detailed explanation, powered by an LLM and retrieval from a medical CSV knowledge base.
- **Easy Reset:** Users can start over at any time.

## How It Works
1. **User Input:** The user describes their main symptom.
2. **Follow-Up Generation:** The app uses a language model and a vector store (built from a CSV of symptoms) to generate relevant follow-up questions.
3. **Answer Collection:** The user answers each follow-up question.
4. **Diagnosis & Explanation:** The app queries the LLM for a likely diagnosis and a reasoning-based explanation.

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd <repo-directory>
```

### 2. Install Dependencies
This project uses a Python virtual environment. Activate it and install the required packages:

```
# If not already in a venv, create one:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Required Python Packages:**
- streamlit
- pandas
- python-dotenv
- langchain
- langchain-community
- chromadb
- openai

You may need to install additional dependencies for Ollama and LLaMA2 support.

### 3. Prepare the Environment
- Create a `.env` file in the project root with your OpenAI API key:
  ```
  OPENAI_API_KEY_4=sk-...
  ```
- Ensure you have a `symptoms_data.csv` file in the project directory. This should contain your medical knowledge base (symptoms and related information).

### 4. Run the App
```
streamlit run app.py
```

The app will open in your browser. Follow the prompts to interact with the diagnostic assistant.

## File Structure
- `app.py` — Main Streamlit application.
- `symptoms_data.csv` — CSV file containing symptom data (required for retrieval).
- `.env` — Environment variables (not included in repo).
- `requirements.txt` — Python dependencies (create if missing).

## Customization
- **Knowledge Base:** Replace or expand `symptoms_data.csv` with your own data for different domains or more comprehensive coverage.
- **LLM Model:** The app uses LLaMA2 via Ollama by default. You can swap in other models supported by LangChain and Ollama.

## Troubleshooting
- Ensure your API keys are correct and the `.env` file is present.
- Make sure all dependencies are installed and the virtual environment is activated.
- If you encounter issues with Ollama or LLaMA2, consult their documentation for setup help.

## License
This project is for educational and research purposes. Please consult a medical professional for real diagnoses.

---

**Enjoy using your AI Diagnostic Assistant!** 
