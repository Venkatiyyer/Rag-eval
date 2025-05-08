import os
import io
import fitz  # PyMuPDF
import faiss

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional , IO
import whisper
import tempfile
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Replaced Chroma with FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile



# Load Whisper Model Once
model = whisper.load_model("tiny.en", device="cpu", download_root="/tmp/whisper")


# Paths to store FAISS indices
PD_PATH = "db/product_docs_faiss"
TP_PATH = "db/top_performers_faiss"



# # ─── Models ───────────────────────────────────────────────────────────────────
# class UploadFeaturesResponse(BaseModel):
#     status: str
#     message: str


# class EvaluateRequest(BaseModel):
#     transcript: str


# class EvaluateResponse(BaseModel):
#     evaluation: str


# class EvaluateAudioResponse(BaseModel):
#     evaluation: str
#     transcript: str  # ✅ Add this line


# ─── Init ─────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup the LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=GROQ_API_KEY
)

# Prompt template for evaluation
prompt_template = PromptTemplate(
    input_variables=["transcript", "top_sample", "product_info"],
    template=""" 
You are an expert evaluator tasked with analyzing a User Transcript for quality and completeness.

Below is the User Transcript:
{transcript}

Below is a Top Performer Sample for comparison:
{top_sample}

Below is the Relevant Product Information:
{product_info}

Your task is to carefully compare the User Transcript against both the Top Performer Sample and the Product Information.

Evaluate the following:
1. How well does the transcript reflect the key product features and benefits?
2. Does it match the tone, structure, and persuasive style of the top performer sample?
3. Are any important product USPs (Unique Selling Propositions) missing or misrepresented?
4. Does the transcript meet an appropriate length for a complete and informative response, or is it too short to be effective?

Provide a comprehensive evaluation with the following structure:

Output:
- **Overall Score**: Out of 10 (Considering -  Content Accuracy: /10, Feature Coverage: /10, Tone & Style: /10, Length Appropriateness: /10)
- **Strengths & Weaknesses**: Summarize what was done well and where it fell short.
- **Missed Product USPs**: Highlight any key product features that were omitted or misrepresented.
- **Improvement Tips**: Offer clear and actionable suggestions to elevate the transcript to top-performer quality.
"""
)

# Embeddings for FAISS
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")


# ─── Utils ────────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[Document]:
    """
    Split text into smaller chunks for processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from uploaded files (PDF or TXT).
    """
    content = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif uploaded_file.name.lower().endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Only PDF or TXT allowed.")



# ─── Ingestion ───────────────────────────────────────────────────────────────
def extract_and_store_in_faiss(product_file, gold_file: Optional[IO] = None) -> dict:
    """Extracts text, stores FAISS indices on disk, and returns extracted text."""
    try:
        # Product docs
        product_text = extract_text_from_file(product_file)
        product_docs = chunk_text(product_text)
        prod_store = FAISS.from_documents(product_docs, embeddings)
        os.makedirs(PD_PATH, exist_ok=True)
        prod_store.save_local(PD_PATH)

        gold_text = None
        if gold_file:
            gold_text = extract_text_from_file(gold_file)
            gold_docs = chunk_text(gold_text)
            gold_store = FAISS.from_documents(gold_docs, embeddings)
            os.makedirs(TP_PATH, exist_ok=True)
            gold_store.save_local(TP_PATH)

        return {"status": "success", "product_text": product_text, "gold_text": gold_text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def evaluate_transcript(transcript: str) -> str:
    """
    Load FAISS indices from disk (created via extract_and_store_in_faiss),
    perform similarity searches, and return LLM evaluation.
    """
    # Verify product index
    if not os.path.isdir(PD_PATH) or not os.listdir(PD_PATH):
        raise RuntimeError(f"Product FAISS index missing at '{PD_PATH}'. Upload products first.")
    product_store = FAISS.load_local(
        PD_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Optionally load gold index
    top_text = ""
    if os.path.isdir(TP_PATH) and os.listdir(TP_PATH):
        top_perf_store = FAISS.load_local(
            TP_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        top_examples = top_perf_store.similarity_search(transcript, k=3)
        top_text = "".join(doc.page_content for doc in top_examples)

    # Product similarity
    prod_examples = product_store.similarity_search(transcript, k=3)
    prod_text = "".join(doc.page_content for doc in prod_examples)

    # Build prompt and invoke LLM
    prompt = prompt_template.format(
        transcript=transcript,
        top_sample=top_text,
        product_info=prod_text
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content



# ─── Evaluate Transcript ──────────────────────────────────────────────────────

def evaluate_transcript(transcript: str) -> str:
    # must always load product store
    product_store = FAISS.load_local(PD_PATH, embeddings, allow_dangerous_deserialization=True)
    prod_examples = product_store.similarity_search(transcript, k=3)
    prod_text = "\n".join(doc.page_content for doc in prod_examples)

    # only load top_perf_store if the gold index exists
    top_text = ""
    if os.path.isdir(TP_PATH) and os.listdir(TP_PATH):
        top_perf_store = FAISS.load_local(TP_PATH, embeddings, allow_dangerous_deserialization=True)
        top_examples   = top_perf_store.similarity_search(transcript, k=3)
        top_text       = "\n".join(doc.page_content for doc in top_examples)

    

    # Build prompt
    prompt = prompt_template.format(
        transcript=transcript,
        top_sample=top_text,
        product_info=prod_text
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


#
# # ─── Evaluate Audio with Real-Time STT ──────────────────────────────────────────
# def evaluate_audio_stt(audio_file) -> dict:
#     """
#     Evaluate an audio file using the real-time STT and perform the same transcript evaluation.
#     """
#     try:
#         audio_bytes = audio_file.read()
#         recorder.set_microphone(False)
#         recorder.feed_audio(audio_bytes)
#         transcript = recorder.text()
#         evaluation = evaluate_transcript(transcript)
#
#         return {"transcript": transcript, "evaluation": evaluation}
#
#     except Exception as e:
#         return {"evaluation": f"Audio Evaluation Failed: {str(e)}"}


def evaluate_audio_whisper(audio_bytes: bytes) -> str:
    try:
        # 1) Write to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 2) Run Whisper transcription
        whisper_result = model.transcribe(tmp_path)
        transcript = whisper_result.get("text", "")

        

        # 3) Clean up temp file
        os.remove(tmp_path)

        # 4) Evaluate the transcript using your existing logic
        evaluation = evaluate_transcript(transcript)
        return {"transcript":transcript, "evaluation":evaluation}

    except Exception as e:
        return {"transcript": "", "evaluation": str(e)}




