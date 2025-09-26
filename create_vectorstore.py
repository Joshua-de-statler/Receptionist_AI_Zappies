import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Verify that the necessary environment variables are set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
     raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

# 1. Load documents from the knowledge_base directory
loader = DirectoryLoader('knowledge_base/', glob="**/*.txt")
docs = loader.load()
if not docs:
    raise ValueError("No documents found in 'knowledge_base' directory. Please add your .txt files.")
print(f"Loaded {len(docs)} documents.")

# 2. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks.")

# 3. Create Google Gemini embeddings
print("Creating Gemini embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. Create a FAISS vector store from the chunks and save it locally
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local("faiss_index")

print("Vector store created and saved locally in 'faiss_index' folder.")