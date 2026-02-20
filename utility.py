from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

'''
Retrieval of info from resume and project files
'''



load_dotenv()
llm_key = os.getenv("LLM_KEY")




embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",api_key=llm_key)
vector_store = Chroma(
    collection_name="user_knowledge",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)

def ingest_pdfs_to_chroma(path: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    all_chunks = []
    all_metadatas = []
    

    reader = PdfReader(path)
    print(f"--- Processing {path} ---")
        
        # Extract text from each page
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

    vector_store.persist()
    print(f"Successfully ingested {len(all_chunks)} chunks into ChromaDB.")

# Usage
# ingest_pdfs_to_chroma("./data/user/Resume.pdf")