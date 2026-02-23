from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

'''
Retrieval of info from resume and project files
'''



load_dotenv()
llm_key = os.getenv("LLM_KEY") or os.getenv("GOOGLE_API_KEY")



embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=llm_key)

def get_vector_store(thread_id: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    persist_path = os.path.join(BASE_DIR, "chroma_db")
    os.makedirs(persist_path, exist_ok=True)
    return Chroma(
    collection_name=f"user_{thread_id}",
    embedding_function=embedding,
    persist_directory=persist_path
    )





def ingest_pdfs_to_chroma(path: str,thread_id: str):
    
    vector_store = get_vector_store(thread_id=thread_id)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    all_chunks = []
    all_metadatas = []
    

    reader = PdfReader(path)
    print(f"--- Processing {path} ---")

    for i, page in enumerate(reader.pages):
        text = page.extract_text()or""
        chunks = text_splitter.split_text(text)
            
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": path, "page": i})


    vector_store.add_texts(
        texts=all_chunks,
        metadatas=all_metadatas
    )
    # Chroma auto-persists when persist_directory is set; .persist() was removed in Chroma 0.4+
    print(f"Successfully ingested {len(all_chunks)} chunks into ChromaDB.")

# Usage
# ingest_pdfs_to_chroma("./data/user/Resume.pdf")