import json
import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Data Preparation
def load_and_prepare_data(manipal_file_path, cdsimer_file_path):
    dataframes = []
    
    # Load Manipal data if available
    try:
        with open(manipal_file_path, 'r') as f:
            manipal_data = json.load(f)
        manipal_df = pd.DataFrame(manipal_data)
        manipal_df['institution'] = 'Manipal Hospitals'
        manipal_df['kmc_reg_no'] = manipal_df.get('kmc_reg_no', 'Not registered')
        dataframes.append(manipal_df)
        st.info(f"Successfully loaded {manipal_file_path}")
    except FileNotFoundError:
        st.warning(f"Manipal data file ({manipal_file_path}) not found. Proceeding with CDSIMER data only.")
    
    # Load CDSIMER data
    try:
        with open(cdsimer_file_path, 'r') as f:
            cdsimer_data = json.load(f)
        cdsimer_df = pd.DataFrame(cdsimer_data)
        cdsimer_df['institution'] = 'CDSIMER'
        cdsimer_df['location'] = cdsimer_df.get('location', 'Not specified')
        cdsimer_df['field_of_expertise'] = cdsimer_df.get('field_of_expertise', cdsimer_df['department'].apply(lambda x: f"Specialist in {x}"))
        dataframes.append(cdsimer_df)
    except FileNotFoundError:
        st.error(f"CDSIMER data file ({cdsimer_file_path}) not found. Cannot proceed without at least one dataset.")
        return None, None
    
    if not dataframes:
        st.error("No datasets loaded. Please ensure at least one valid dataset is provided.")
        return None, None
    
    # Combine datasets
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Normalize and fill missing fields
    for df in [combined_df]:
        df['qualification'] = df['qualification'].fillna('Not specified')
        df['kmc_reg_no'] = df['kmc_reg_no'].fillna('Not registered')
        df['field_of_expertise'] = df['field_of_expertise'].fillna(df['department'].apply(lambda x: f"Specialist in {x}"))
        df['department'] = df['department'].str.lower().str.replace('clincal', 'clinical')
        df['location'] = df['location'].fillna('Not specified')
    
    # Combine relevant fields for embedding
    combined_df['combined_text'] = combined_df.apply(
        lambda row: f"{row['name']} is a {row['designation']} in {row['department']} at {row['institution']} ({row['location']}). "
                    f"Qualifications: {row['qualification']}. Expertise: {row['field_of_expertise']}.",
        axis=1
    )
    
    # Create documents for LangChain
    documents = [
        Document(
            page_content=row['combined_text'],
            metadata={
                'name': row['name'],
                'designation': row['designation'],
                'department': row['department'],
                'institution': row['institution'],
                'location': row['location'],
                'qualification': row['qualification'],
                'field_of_expertise': row['field_of_expertise'],
                'kmc_reg_no': row['kmc_reg_no'],
                'source_url': row['source_url']
            }
        )
        for _, row in combined_df.iterrows()
    ]
    
    return documents, combined_df

# 2. Setup Embedding Model and Vector Store
def setup_vector_store(documents):
    # Load HuggingFace embeddings
    embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embedder)
    
    # Save vector store locally
    vector_store.save_local("faiss_index")
    
    return vector_store, embedder

# 3. Setup Language Model
def setup_llm():
    # Using deepseek-r1:1.5b via Ollama
    llm = Ollama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.7,
        num_predict=200
    )
    return llm

# 4. Setup RAG Pipeline
def setup_rag_pipeline(vector_store, llm):
    prompt_template = """
    You are a medical assistant. Based on the following doctor profiles from Manipal Hospitals and CDSIMER, answer the query concisely and accurately.
    Query: {question}
    
    Doctor Profiles:
    {context}
    
    Provide a clear response listing relevant doctors, their department, designation, expertise, institution, and location. Include source URLs if available.
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# 5. Extract Department and Check for "All" Keywords
def extract_department_and_all(query, departments):
    all_keywords = ["all", "every", "entire"]
    query_lower = query.lower()
    
    # Check for "all" keywords
    has_all = any(keyword in query_lower for keyword in all_keywords)
    
    # Extract department
    department = None
    for dept in departments:
        if dept.lower() in query_lower:
            department = dept.lower()
            break
    
    return has_all, department

# 6. Retrieve All Documents for a Department
def get_all_department_docs(documents, department):
    return [doc for doc in documents if doc.metadata['department'].lower() == department]

# 7. Streamlit UI
def main():
    st.set_page_config(page_title="Doctor Search: ", layout="wide")
    st.title("Doctor Search")
    st.write("Search for doctors by department, designation, expertise, or location (e.g., 'all doctors in pediatrics' or 'cardiologist in Jayanagar').")
    
    # Define file paths
    manipal_file_path = r"E:\\workspace\\CDSIMER\\Faculty Details\\manipal_faculty_data.json"
    cdsimer_file_path = r"E:\\workspace\\CDSIMER\\Faculty Details\\faculty_data.json"
    
    # Load data and setup pipeline
    documents, combined_df = load_and_prepare_data(manipal_file_path, cdsimer_file_path)
    
    if documents is None or combined_df is None:
        st.stop()
    
    if not os.path.exists("faiss_index"):
        vector_store, embedder = setup_vector_store(documents)
    else:
        embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    
    llm = setup_llm()
    qa_chain = setup_rag_pipeline(vector_store, llm)
    
    # Get list of departments for query parsing
    departments = combined_df['department'].unique()
    
    # User input
    query = st.text_input("Enter your query ")
    
    if query:
        with st.spinner("Searching for doctors..."):
            # Check for "all" and department
            has_all, department = extract_department_and_all(query, departments)
            
            if has_all and department:
                # Retrieve all documents for the department
                retrieved_docs = get_all_department_docs(documents, department)
                # Run RAG pipeline for generated response
                response = qa_chain.run(query)
            else:
                # Use FAISS similarity search for other queries
                retrieved_docs = vector_store.similarity_search(query, k=5)
                response = qa_chain.run(query)
        
        # st.subheader("Generated Response")
        # st.write(response)
        
        # Display raw retrieved documents
        st.subheader("Retrieved Doctor Profiles")
        if not retrieved_docs:
            st.write("No matching profiles found.")
        else:
            for doc in retrieved_docs:
                st.write("---")
                st.write(f"**Name**: {doc.metadata['name']}")
                st.write(f"**Designation**: {doc.metadata['designation']}")
                st.write(f"**Department**: {doc.metadata['department'].capitalize()}")
                st.write(f"**Institution**: {doc.metadata['institution']}")
                st.write(f"**Location**: {doc.metadata['location']}")
                st.write(f"**Expertise**: {doc.metadata['field_of_expertise']}")
                st.write(f"**Qualifications**: {doc.metadata['qualification']}")
                st.write(f"**KMC Registration No**: {doc.metadata['kmc_reg_no']}")
                st.write(f"[Source URL]({doc.metadata['source_url']})")

if __name__ == "__main__":
    main()