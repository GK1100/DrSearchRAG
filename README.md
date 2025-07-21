# DrSearchRAG

A Retrieval-Augmented Generation (RAG) system built with LangChain and Streamlit to search and retrieve doctor profiles from Manipal Hospitals and CDSIMER datasets. It uses HuggingFace embeddings and a local FAISS vector store for efficient retrieval combined with an LLM for answer generation.

🚀 Features
Search doctors by department, designation, expertise, or location

Uses FAISS vector store with sentence-transformers/all-MiniLM-L6-v2 embeddings

Streamlit UI for interactive queries

Integrated with Ollama LLM (deepseek-r1:1.5b)

Retrieves and displays detailed doctor profiles

Supports "all" queries to list every doctor in a department

🛠️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/GK1100/DrSearchRAG.git
cd doctor-search-rag


💻 Usage
Run the Streamlit app:

bash
Copy
Edit
streamlit run main.py
Then open the URL displayed in your terminal (usually http://localhost:8501) to interact with the UI.

📝 Example Queries
"All doctors in pediatrics"

"Cardiologist in Jayanagar"

"Who is the head of neurology?"
