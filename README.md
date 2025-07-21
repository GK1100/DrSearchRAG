# DrSearchRAG

A **Streamlit-based web application** that allows users to search for doctor profiles from Manipal Hospitals and CDSIMER. The app leverages modern NLP toolkits (LangChain, HuggingFace, FAISS) and Retrieval-Augmented Generation (RAG) to deliver relevant results for medical professional queries.

## Features

- **Search by department, designation, expertise, or location**
- Hybrid search: combines semantic vector search with retrieval-augmented answers
- Supports complex queries like "all doctors in pediatrics" or "cardiologist in Jayanagar"
- Profiles display include: name, designation, department, qualification, expertise, institution, location, KMC registration, and source URL


## Tech Stack

- **Python** (Pandas, Streamlit)
- **LangChain** for LLM-driven RAG pipeline
- **HuggingFace sentence-transformers** for embeddings
- **FAISS** for fast, scalable vector search
- **Ollama** with the `deepseek-r1:1.5b` model (local deployment)


## Setup Instructions

1. **Clone this repository:**

```bash
git clone <https://github.com/GK1100/DrSearchRAG.git>
```

2. **Start Ollama server and download the desired model:**
    - Ensure `Ollama` runs at `http://localhost:11434` and the `deepseek-r1:1.5b` model is available.
5. **Run the Streamlit app:**

```bash
streamlit run main.py
```


## Usage

- Open the web UI via the URL displayed in your terminal (typically http://localhost:8501).
- Enter queries like:
    - `"all doctors in pediatrics"`
    - `"endocrinologist in Bangalore"`
    - `"professor of surgery"`

## Notes

- Large data or model files are not tracked by git for repository size constraints.
- Some features (like FAISS index export) require sufficient disk space.
- Ollama and the specified model must be running locally before starting the app.


## License

This project is licensed under the MIT License.

<div style="text-align: center">⁂</div>

[^1]: main.py


