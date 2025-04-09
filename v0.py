import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Load Gemini API key from environment variable
# Ensure you have set the GOOGLE_API_KEY environment variable
# For testing, you can temporarily set it here (but avoid in production)
os.environ['GOOGLE_API_KEY'] = "AIzaSyDNVP2zL4Fvv5dA0bjo4kN-rszIM5VDNlk"

# Initialize Gemini Pro model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def extract_pdf_in_markdown():
    import pymupdf4llm
    md_text = pymupdf4llm.to_markdown("security_document.pdf")
    return md_text
    # import pathlib
    # pathlib.Path("re.md").write_bytes(md_text.encode())


# # ---------- STEP 2: Chunk Text ----------
# def chunk_text(text, chunk_size=500, chunk_overlap=100):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     return splitter.split_text(text)
#
#
# # ---------- STEP 3: Convert to LangChain Documents ----------
# def create_documents(chunks, source="source-pdf"):
#     return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

headers_to_split_on = [
    ("#", "heading1"),
    ("##", "heading2"),
    ("###", "heading3"),
]


def smart_chunk_markdown(md_text):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)
    return docs  # returns LangChain Document objects directly


# ---------- STEP 4: Embed with HuggingFace & Store in FAISS ----------
def create_faiss_index(documents, faiss_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore


def create_faiss_index_from_markdown(md_text, faiss_path):
    extract_pdf_in_markdown()
    docs = smart_chunk_markdown(md_text)
    return create_faiss_index(docs, faiss_path)


# ---------- STEP 5: Load FAISS + Search + Use Gemini ----------
def query_faiss_and_respond(query, faiss_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    relevant_docs = db.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Gemini LLM call (RAG)
    prompt = f"""You are a UPSC study assistant. Based on the following context, generate a concise and structured response.

Context:
{context}

Question:
{query}

Answer:"""

    response = llm.invoke(prompt)
    return response.text


# create_faiss_index_from_markdown(extract_pdf_in_markdown(), "faiss_index/security_index")
response = query_faiss_and_respond(
    " please explain Data Protection",
    "faiss_index/security_index"
)
print(response)
