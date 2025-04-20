from fastapi import FastAPI, UploadFile, Form, Depends, Path
from fastapi.responses import JSONResponse
import os

from utils.supabase_storage import SupabaseStorage
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import pymupdf4llm
import shutil
from fastapi import Request, Form
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from crud import create_pdf_index, increment_api_request_count, create_mcq_response, get_all_pdf_indices_for_institution
from database import engine, get_db
import models
from routers import pdf_index, institutions, mcq_response
from schemas import PDFIndexCreate, MCQResponseCreate

app = FastAPI()
models.Base.metadata.create_all(bind=engine)
app.include_router(pdf_index.router)
app.include_router(institutions.router)
app.include_router(mcq_response.router)


# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

headers_to_split_on = [
    ("#", "heading1"),
    ("##", "heading2"),
    ("###", "heading3"),
]


def extract_pdf_in_markdown(file_path: str) -> str:
    return pymupdf4llm.to_markdown(file_path)


def smart_chunk_markdown(md_text):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(md_text)


def create_faiss_index(documents, faiss_path):
    os.makedirs(faiss_path, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore


def load_faiss_index(faiss_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)


# Configure CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://civisync.netlify.app"
    # Add any other origins you need, e.g. your deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ... your routes below ...


@app.post("/{institution_id}/upload-pdf/")
async def upload_pdf(file: UploadFile,
                     index_name: str = Form(...),
                     db: Session = Depends(get_db),
                     institution_id: str = Path(...)
                     ):
    try:
        # Save the uploaded file temporarily
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract and index content
        md_text = extract_pdf_in_markdown(temp_path)
        documents = smart_chunk_markdown(md_text)
        index_dir_path = f"faiss_index/{index_name}"
        os.makedirs("faiss_index", exist_ok=True)
        create_faiss_index(documents, index_dir_path)

        # Upload to Supabase
        supabase = SupabaseStorage()
        index_path = f"{index_dir_path}/index.faiss"
        pkl_path = f"{index_dir_path}/index.pkl"
        index_supabase_path = f"{institution_id}/{index_path}"
        pkl_supabase_path = f"{institution_id}/{pkl_path}"
        supabase_path = f"{institution_id}/{file.filename}"
        public_path = supabase.upload_file(temp_path, supabase_path)
        supabase.upload_file(index_path, index_supabase_path)
        supabase.upload_file(pkl_path, pkl_supabase_path)

        # Save metadata to DB
        pdf_index_data = PDFIndexCreate(
            index_name=index_name,
            filename=public_path,
            institution_id=institution_id
        )
        create_pdf_index(db=db, pdf_data=pdf_index_data)

        # Cleanup local file
        try:
            os.remove(index_path)
            os.remove(pkl_path)
            os.rmdir(index_dir_path)
            os.remove(temp_path)
            os.rmdir("faiss_index")
            os.rmdir(temp_dir)
        except OSError:
            pass
        return JSONResponse(content={"message": "Index created successfully.", "file": public_path})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/{institution_id}/query/")
async def query_index(
        query: str, index_name: str,
        db: Session = Depends(get_db),
        content_type: str = "MCQ",
        institution_id: str = Path(...)
):
    try:
        index_path = f"faiss_index/{index_name}"

        supabase = SupabaseStorage()
        index_supabase_path = f"{institution_id}/{index_path}/index.faiss"
        pkl_supabase_path = f"{institution_id}/{index_path}/index.pkl"

        index_local_path = os.path.join(index_path, "index.faiss")
        pkl_local_path = os.path.join(index_path, "index.pkl")

        supabase.download_file(index_supabase_path, index_local_path)
        supabase.download_file(pkl_supabase_path, pkl_local_path)

        f_db = load_faiss_index(index_path)
        relevant_docs = f_db.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        mcq_prompt = create_prompt(query, context, content_type)

        # Create prompt
        response = llm.invoke(mcq_prompt)

        increment_api_request_count(
            db=db,
            institution_id=institution_id,
        )

        pdf_index = db.query(models.PDFIndex).filter(
            index_name == index_name,
            institution_id == institution_id
        ).first()

        # Save PDF index metadata to DB
        mcq_data = MCQResponseCreate(
            institution_id = institution_id,
            pdf_index_id = pdf_index.id,
            mcq_data = response.content
        )

        create_mcq_response(
            db = db,
            mcq_data = mcq_data
        )
        try:
            os.remove(index_local_path)
            os.remove(pkl_local_path)
            os.rmdir(index_path)
            os.rmdir("faiss_index")
        except OSError:
            pass
        return JSONResponse(content={"response": response.content})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/{institution_id}/query-ollama/")
async def query_with_ollama(
        query: str,
        index_name: str,
        content_type: str = "MCQ",
        db: Session = Depends(get_db),
        institution_id: str = Path(...)
):
    try:
        index_path = f"faiss_index/{index_name}"
        supabase = SupabaseStorage()
        index_supabase_path = f"{institution_id}/{index_path}/index.faiss"
        pkl_supabase_path = f"{institution_id}/{index_path}/index.pkl"

        index_local_path = os.path.join(index_path, "index.faiss")
        pkl_local_path = os.path.join(index_path, "index.pkl")

        supabase.download_file(index_supabase_path, index_local_path)
        supabase.download_file(pkl_supabase_path, pkl_local_path)

        # Setup Ollama components
        model_name = "llama3.2"  # or whatever Ollama model you're using
        ollma = OllamaLLM(model=model_name)
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Get relevant documents
        docs = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        # Create prompt
        mcq_prompt = create_prompt(query, context, content_type)

        # Call Ollama LLM
        response = ollma.invoke(mcq_prompt)

        increment_api_request_count(
            db=db,
            institution_id=institution_id
        )

        pdf_index = db.query(models.PDFIndex).filter(
            index_name == index_name,
            institution_id == institution_id
        ).first()

        # Save PDF index metadata to DB
        mcq_data = MCQResponseCreate(
            institution_id = institution_id,
            pdf_index_id = pdf_index.id,
            mcq_data = response
        )

        create_mcq_response(
            db = db,
            mcq_data = mcq_data
        )

        try:
            os.remove(index_local_path)
            os.remove(pkl_local_path)
            os.rmdir(index_path)
            os.rmdir("faiss_index")
        except OSError:
            pass

        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

PDF_DIR = "temp_files"
INDEX_DIR = "faiss_index"


@app.get("/{institution_id}/list-resources/")
async def list_resources(
        institution_id: str = Path(...),
        db: Session = Depends(get_db)
):
    pdf_indices = get_all_pdf_indices_for_institution(
        db=db,
        institution_id=institution_id
    )
    # List FAISS indices (directories under faiss_index)
    indices = []
    if os.path.isdir(INDEX_DIR):
        indices = [d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))]

    return JSONResponse(content={
        "pdfs": [pdf.filename for pdf in pdf_indices],
        "indices": [pdf.index_name for pdf in pdf_indices]
        # "pdfs": indices,
        # "indices": indices
    })


def create_prompt(
        query: str,
        context: str,
        prompt_type: str = "MCQ"
):
    if prompt_type == "MCQ":
        return f"""
               You are an expert UPSC Prelims question setter with deep knowledge of the UPSC syllabus and exam pattern. 

               Given the following study material (context), generate **{query}** in the style of the UPSC Civil Services Preliminary Examination. Each question should:

               1. **Be statement-based**, requiring candidates to choose the most appropriate option.
               2. Have **four options** labeled (A), (B), (C), and (D).
               3. Indicate the **correct answer** clearly.
               4. Provide a **brief explanation** (1–2 sentences) justifying the correct option.
               5. Optionally tag the question with a **difficulty level** (Easy, Medium, Hard).

               ### Context:
               {context}

               ### Instructions:
               - {query}
               - Do **not** reveal the explanation before listing the questions.
               - After the last question, list the answers and explanations in a separate section titled “Answers and Explanations.”
               - Maintain formal, concise language suitable for UPSC aspirants.

               ### Output Format (Markdown):
               ```markdown
               1. Question statement?

                  (A) Option A  
                  (B) Option B  
                  (C) Option C  
                  (D) Option D  

               2. …

               ---

               **Answers and Explanations**

               1. **Answer:** (B)  
                  **Explanation:** Brief justification here.

               2. **Answer:** (C)  
                  **Explanation:** …

               …

               Given following examples of MCQs do not output the same MCQs. this are the only for reference
               1. Question1 points
               WWhich of the following Articles of the Indian Constitution deal with the Directive Principles of State Policy?

                  (A) Articles 36–51  
                  (B) Articles 39–51  
                  (C) Articles 40–50  
                  (D) Articles 42–52  

               2. Question1 points
               Consider the following statements:

               Fungi are considered prokaryotic organisms due to the absence of a nucleus.
               Lichens are made solely of fungal species.
               Some fungi have shown the ability to degrade synthetic plastic materials.
               How many of the above statements is/are correct?

                (a) Only one
                (b) Only two
                (c) All three
                (d) None
               3. Question1 points
               Consider the following statements:

               Strike-slip faults always produce vertical displacement of land.
               The Indian Plate is moving southward relative to the Eurasian Plate.
               The epicentre is the point of origin of the earthquake beneath the Earth’s surface.
               How many of the above statements is/are correct?

                (a) Only one
                (b) Only two
                (c) All three
                (d) None
               4. Question1 points
               Which of the following best explains why photonic chips are considered superior for future technologies like 6G and quantum computing?

                (a) They require no electricity for operation
                (b) They rely solely on quantum bits (qubits) instead of classical bits
                (c) They offer ultra-high data transfer speeds and minimal heat generation
                (d) They can only be used in space applications
               5. Question1 points
               Consider the following statements regarding Gene banks.

               Statement-I: Gene banks are crucial for India’s climate-resilient agriculture.
               Statement-II: Gene banks provide immediate high-yield varieties to farmers.

               Which one of the following is correct in respect of the above statements?

                a) Both Statement-I and Statement-II are correct and Statement-II is the correct explanation for Statement-I
                b) Both Statement-I and Statement-II are correct and Statement-II is not the correct explanation for Statement-I
                c) Statement-I is correct but Statement-II is incorrect
                d) Statement-I is incorrect but Statement-II is correct
               6. Question1 points
               Consider the following statements about the practice of light fishing:
               1. It is most commonly used in the monsoon season when fish abundance is at its peak.
               2. It can interfere with the natural reproductive cycles of marine species.
               3. It increases the risk of biodiversity loss by attracting non-target marine species.
               Which of the above statements are correct?

                a) 1 and 2 only
                b) 2 and 3 only
                c) 1 and 3 only
                d) 1, 2 and 3

               7. Question1 points
               Consider the following statements regarding Parker Solar Probe

               Statement-I: Parker Solar Probe helps improve forecasting of space weather.
               Statement-II: Space weather has minimal effect on satellite communications and power grids on Earth.

               Which one of the following is correct?

                a) Both Statement-I and Statement-II are correct and Statement-II is the correct explanation for Statement-I
                b) Both Statement-I and Statement-II are correct and Statement-II is not the correct explanation for Statement-I
                c) Statement-I is correct but Statement-II is incorrect
                d) Statement-I is incorrect but Statement-II is correct
               """
    elif prompt_type == "ESSAY":
        return f"""
                You are an expert UPSC essay writer with deep knowledge of the UPSC syllabus and exam format.

                Given the following context, generate essay format and sample content ideas relevant for the UPSC Civil
                 Services Examination. Ensure topics are and for each topic generate short summary for essay writing:
                
                1. Thought-provoking and aligned with current affairs, ethics, and governance.
                2. Broadly inclusive of socio-economic, political, and international dimensions.
                3. Suitable for essay writing practice in UPSC Mains.

                ### Context:
                {context}

                ### Instructions:
                - {query}
                - Provide a list of essay topics and structured content with headings/subheadings.
                - Maintain concise, formal language appropriate for UPSC aspirants.
                - Format should be strictly Use Markdown.(MUST BE MARKDOWN)

                ### Output Format:
                ```markdown
                1. Essay Topic 1: Title
                   - Brief outline and related content.
                2. Essay Topic 2: Title
                 - Brief outline and related content.
                ...
                ```
                """
    elif prompt_type == "NOTES":
        return f"""
                You are an expert in creating concise yet comprehensive study notes for UPSC aspirants based on the provided context.

                Given the following context, generate structured notes in bullet points. The notes should:
                
                1. Cover key concepts, facts, and figures.
                2. Be organized under relevant headings and subheadings.
                3. Highlight important points in bold.
                4. Include examples where possible.
                5. Format should be strictly Use Markdown.(MUST BE MARKDOWN)

                ### Context:
                {context}

                ### Instructions:
                - {query}
                - Ensure readability and clarity for exam preparation.
                - Use bullet points and maintain brevity.
                
                ### Output Format:
                ```markdown
                ## Heading 1
                - Point 1
                - Point 2

                ## Heading 2
                - Point 1
                - Example: …
                ```
                """
    else:
        return f"""
                You are an expert content generator for UPSC preparation material with deep expertise in the syllabus and exam pattern.

                Based on the provided study material (context), generate **{query}** suitable for UPSC aspirants. Ensure the content is:
                
                1. Well-structured and concise.
                2. Aligned with the UPSC exam requirements.
                3. Relevant to current affairs, syllabus topics, or exam trends.

                ### Context:
                {context}

                ### Instructions:
                - {query}
                - Maintain formal, precise language.
                - Use Markdown formatting for organization where applicable.
                
                Output content relevant for the UPSC exam.
                """
