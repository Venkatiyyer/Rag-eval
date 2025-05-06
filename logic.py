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
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from RealtimeSTT import AudioToTextRecorder

# ─── Load Whisper Model ──────────────────────────────────────────────────────
model = whisper.load_model("tiny.en")

# ─── Models ──────────────────────────────────────────────────────────────────
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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

top_perf_store = Chroma(
    embedding_function=embeddings,
    collection_name="top_performers",
    persist_directory="db/top_performers"
)
top_perf_retriever = top_perf_store.as_retriever(search_kwargs={"k": 3})

product_store = Chroma(
    embedding_function=embeddings,
    collection_name="product_docs",
    persist_directory="db/product_docs"
)
product_retriever = product_store.as_retriever(search_kwargs={"k": 3})


# ─── Utils ────────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


async def extract_text_from_file(uploaded) -> str:
    content = await uploaded.read()
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


# ─── Functional Versions of Former Endpoints ──────────────────────────────────
async def upload_product_features_fn(product_file, gold_file=None) -> dict:
    try:
        product_text = await extract_text_from_file(product_file)
        product_docs = chunk_text(product_text)
        product_store.add_documents(product_docs)
        product_store.persist()

        if gold_file:
            gold_text = await extract_text_from_file(gold_file)
            gold_docs = chunk_text(gold_text)
            top_perf_store.add_documents(gold_docs)
            top_perf_store.persist()

        return {"status": "success", "message": "Product doc uploaded." + (" Gold example added." if gold_file else "")}

    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}


def evaluate_fn(data: EvaluateRequest) -> dict:
    transcript = data.transcript
    evaluation = evaluate_transcript(transcript)
    return {"evaluation": evaluation}


async def evaluate_audio_stt_fn(audio_file) -> dict:
    try:
        audio_bytes = await audio_file.read()
        recorder.set_microphone(False)
        recorder.feed_audio(audio_bytes)
        transcript = recorder.text()
        evaluation = evaluate_transcript(transcript)
        return {"transcript": transcript, "evaluation": evaluation}
    except Exception as e:
        return {"evaluation": str(e)}


async def evaluate_audio_whisper_fn(audio_file) -> dict:
    try:
        audio_bytes = await audio_file.read()
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
