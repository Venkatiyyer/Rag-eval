import os
import io
import fitz  # PyMuPDF
import faiss

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import whisper
import tempfile
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Replaced Chroma with FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from RealtimeSTT import AudioToTextRecorder  # Import your real-time STT recorder

# ─── Load Whisper Model Once ─────────────────────────────────────────────────
# model = whisper.load_model("tiny.en")  # or "base", "medium", "large"


# ─── Models ───────────────────────────────────────────────────────────────────
class UploadFeaturesResponse(BaseModel):
    status: str
    message: str


class EvaluateRequest(BaseModel):
    transcript: str


class EvaluateResponse(BaseModel):
    evaluation: str


class EvaluateAudioResponse(BaseModel):
    evaluation: str
    transcript: str  # ✅ Add this line


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
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ─── Helpers ─────────────────────────────────────────────────────────────────
DB_ROOT = "db/faiss"
TP_PATH = os.path.join(DB_ROOT, "top_performers")
PD_PATH = os.path.join(DB_ROOT, "product_docs")

# ─── Utils ────────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[Document]:
    """
    Split text into smaller chunks for processing.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


def extract_text_from_file(uploaded) -> str:
    """
    Extract text from uploaded files (PDF or TXT).
    """
    content = uploaded.read()
    if uploaded.filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif uploaded.filename.lower().endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Only PDF or TXT allowed.")


# ─── Upload Product + Gold Example ──────────────────────────────────────────────
def upload_product_features(product_file, gold_file=None) -> dict:
    try:
        # 1) ingest product docs
        product_text = extract_text_from_file(product_file)
        product_docs = chunk_text(product_text)
        pd_store = FAISS.from_documents(product_docs, embeddings)
        os.makedirs(PD_PATH, exist_ok=True)
        pd_store.save_local(PD_PATH)

        msg = "Product doc uploaded."
        # 2) ingest gold sample if provided
        if gold_file:
            gold_text = extract_text_from_file(gold_file)
            gold_docs = chunk_text(gold_text)
            tp_store = FAISS.from_documents(gold_docs, embeddings)
            os.makedirs(TP_PATH, exist_ok=True)
            tp_store.save_local(TP_PATH)
            msg += " Gold example added."

        return {"status": "success", "message": msg}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Evaluate Transcript ──────────────────────────────────────────────────────

def evaluate_transcript(transcript: str) -> str:
    # 1) load the two persisted stores
    tp_store = FAISS.load_local(TP_PATH, embeddings)
    pd_store = FAISS.load_local(PD_PATH, embeddings)
    # 2) retrieve top‐k
    top_docs  = tp_store.similarity_search(transcript, k=3)
    prod_docs = pd_store.similarity_search(transcript, k=3)
    # 3) call LLM
    prompt = prompt_template.format(
        transcript=transcript,
        top_sample="\n".join(d.page_content for d in top_docs),
        product_info="\n".join(d.page_content for d in prod_docs),
    )
    return llm.invoke([{"role": "user", "content": prompt}]).content


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


# ─── Evaluate Audio with Whisper ───────────────────────────────────────────────
def evaluate_audio_whisper(audio_file) -> dict:
    """
    Evaluate an audio file using Whisper and perform the same transcript evaluation.
    """
    try:
        model = whisper.load_model("tiny.en")
        audio_bytes = audio_file.read()

        # Write the audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe using Whisper
        whisper_result = model.transcribe(tmp_path)
        transcript = whisper_result["text"]

        # Clean up the temporary file
        os.remove(tmp_path)

        # Evaluate the transcript
        evaluation = evaluate_transcript(transcript)
        return {"transcript": transcript, "evaluation": evaluation}

    except Exception as e:
        return {"evaluation": f"Audio Evaluation Failed: {str(e)}"}


# ─── Initialize Real-Time STT Recorder ─────────────────────────────────────────
recorder = AudioToTextRecorder(
    spinner=False,
    silero_sensitivity=0.01,
    model="tiny.en",
    language="en"  # Disable continuous mic capture
)

# Immediately turn its mic off so it won’t buffer live audio
recorder.set_microphone(False)

