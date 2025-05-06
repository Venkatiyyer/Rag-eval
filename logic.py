# ─── Imports ──────────────────────────────────────────────────────────────────
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pydantic import BaseModel
import io
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from typing import List, Optional
import whisper
import tempfile
import speech_recognition as sr

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # ✅ Switched to FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from RealtimeSTT import AudioToTextRecorder
from sentence_transformers import SentenceTransformer

# ─── Embedding Model ─────────────────────────────────────────────────────────
model_1 = SentenceTransformer("all-MiniLM-L6-v2")
model_1.save("local_models/all-MiniLM-L6-v2")

# ─── Load Whisper Model ──────────────────────────────────────────────────────
model = whisper.load_model("tiny.en")

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class UploadFeaturesResponse(BaseModel):
    status: str
    message: str

class EvaluateRequest(BaseModel):
    transcript: str

class EvaluateResponse(BaseModel):
    evaluation: str

class EvaluateAudioResponse(BaseModel):
    evaluation: str
    transcript: str

# ─── Init ─────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

prompt_template = PromptTemplate(
    input_variables=["transcript", "top_sample", "product_info"],
    template="""..."""  # Unchanged prompt template body
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── Helper Functions to Load Documents ───────────────────────────────────────

# ─── Load Top Performer Documents Using BytesIO ──────────────────────────────
def load_top_perf_docs(file_data: bytes) -> List[Document]:
    """
    Load top performer documents from a file received as in-memory bytes.
    The file_data (bytes) could be a .txt or .pdf file.
    """
    docs = []
    
    # Use BytesIO to handle the in-memory file
    file_stream = io.BytesIO(file_data)
    
    # Determine file type (we'll assume either .txt or .pdf)
    if file_stream.getbuffer().startswith(b"%PDF"):  # Check if PDF file
        pdf = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        docs.append(Document(page_content=text, metadata={"source": "top_performer.pdf"}))

    else:
        # If it's not a PDF, we assume it's a .txt file
        text = file_stream.read().decode("utf-8")
        docs.append(Document(page_content=text, metadata={"source": "top_performer.txt"}))

    return docs

# ─── Load Product Documents Using BytesIO ────────────────────────────────────
def load_product_docs(file_data: bytes) -> List[Document]:
    """
    Load product documents from a file received as in-memory bytes.
    The file_data (bytes) could be a .txt or .pdf file.
    """
    docs = []
    
    # Use BytesIO to handle the in-memory file
    file_stream = io.BytesIO(file_data)
    
    # Determine file type (we'll assume either .txt or .pdf)
    if file_stream.getbuffer().startswith(b"%PDF"):  # Check if PDF file
        pdf = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        docs.append(Document(page_content=text, metadata={"source": "product_doc.pdf"}))

    else:
        # If it's not a PDF, we assume it's a .txt file
        text = file_stream.read().decode("utf-8")
        docs.append(Document(page_content=text, metadata={"source": "product_doc.txt"}))

    return docs

# ─── FAISS Vector Store Initialization Without Loading ──────────────────────

# Function to initialize the top performers store
def initialize_top_perf_store() -> FAISS:
    # Load your top performer documents
    top_perf_docs = load_top_perf_docs()  # You need to implement this function
    return FAISS.from_documents(top_perf_docs, embeddings)

# Function to initialize the product store
def initialize_product_store() -> FAISS:
    # Load your product documents
    product_docs = load_product_docs()  # You need to implement this function
    return FAISS.from_documents(product_docs, embeddings)

# Initialize the stores (this will run when the application starts or when needed)
top_perf_store = initialize_top_perf_store()
top_perf_retriever = top_perf_store.as_retriever(search_kwargs={"k": 3})

product_store = initialize_product_store()
product_retriever = product_store.as_retriever(search_kwargs={"k": 3})

# ─── Utils ────────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

def extract_text_from_file(uploaded) -> str:
    content = uploaded.read()
    if uploaded.filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif uploaded.filename.lower().endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Only PDF or TXT allowed.")

def evaluate_transcript(transcript: str) -> str:
    top_examples = top_perf_retriever.get_relevant_documents(transcript)
    top_text = "\n".join(doc.page_content for doc in top_examples)
    prod_examples = product_retriever.get_relevant_documents(transcript)
    prod_text = "\n".join(doc.page_content for doc in prod_examples)
    prompt = prompt_template.format(
        transcript=transcript,
        top_sample=top_text,
        product_info=prod_text,
    )
    llm_resp = llm.invoke([{"role": "user", "content": prompt}])
    return llm_resp.content

# ─── Upload and Evaluate ──────────────────────────────────────────────────────
def upload_product_features_fn(product_file, gold_file=None) -> dict:
    try:
        product_text = extract_text_from_file(product_file)
        product_docs = chunk_text(product_text)
        product_store = FAISS.from_documents(product_docs, embeddings)
        product_store.save_local("db/product_docs")

        if gold_file:
            gold_text = extract_text_from_file(gold_file)
            gold_docs = chunk_text(gold_text)
            top_perf_store = FAISS.from_documents(gold_docs, embeddings)
            top_perf_store.save_local("db/top_performers")

        return {"status": "success", "message": "Product doc uploaded." + (" Gold example added." if gold_file else "")}

    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

def evaluate_fn(data: EvaluateRequest) -> dict:
    transcript = data.transcript
    evaluation = evaluate_transcript(transcript)
    return {"evaluation": evaluation}

def evaluate_audio_stt_fn(audio_file) -> dict:
    try:
        audio_bytes = audio_file.read()
        recorder.set_microphone(False)
        recorder.feed_audio(audio_bytes)
        transcript = recorder.text()
        evaluation = evaluate_transcript(transcript)
        return {"transcript": transcript, "evaluation": evaluation}
    except Exception as e:
        return {"evaluation": str(e)}

def evaluate_audio_whisper_fn(audio_file) -> dict:
    try:
        audio_bytes = audio_file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        whisper_result = model.transcribe(tmp_path)
        transcript = whisper_result["text"]
        os.remove(tmp_path)

        evaluation = evaluate_transcript(transcript)
        return {"transcript": transcript, "evaluation": evaluation}
    except Exception as e:
        return {"evaluation": f"Audio Evaluation Failed: {str(e)}"}

# ─── Initialize STT ───────────────────────────────────────────────────────────
recorder = AudioToTextRecorder(
    spinner=False,
    silero_sensitivity=0.01,
    model="tiny.en",
    language="en"
)
recorder.set_microphone(False)

