from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import io

import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from typing import List
import whisper
import tempfile
from pathlib import Path


from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Load Whisper Model Once ─────────────────────────────────────────────────
model = None
# Replace with more memory-efficient loading
#model = whisper.load_model("tiny.en", device="cpu", download_root="/tmp/whisper")


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

app = FastAPI()

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

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")


# # ─── Vector Stores ────────────────────────────────────────────────────────────
# # Always load existing FAISS indices; they must be created via /upload_product_features first
# # Allow pickle deserialization since you trust your own data
#
# top_perf_store = FAISS.load_local("db/top_performers_faiss", embeddings,allow_dangerous_deserialization=True)
# top_perf_retriever = top_perf_store.as_retriever(search_kwargs={"k": 3})
#
# product_store = FAISS.load_local("db/product_docs_faiss", embeddings, allow_dangerous_deserialization=True)
# product_retriever = product_store.as_retriever(search_kwargs={"k": 3})

# Vector db var initialisation

top_perf_store = None
top_perf_retriever = None
product_store = None
product_retriever = None

# helper function for lazy loading:

# def load_vector_stores():
#     global top_perf_store, top_perf_retriever, product_store, product_retriever
#     if top_perf_store is None and Path("db/top_performers_faiss/index.faiss").exists():
#         top_perf_store = FAISS.load_local("db/top_performers_faiss", embeddings, allow_dangerous_deserialization=True)
#         top_perf_retriever = top_perf_store.as_retriever(search_kwargs={"k": 3})
#
#     if product_store is None and Path("db/product_docs_faiss/index.faiss").exists():
#         product_store = FAISS.load_local("db/product_docs_faiss", embeddings, allow_dangerous_deserialization=True)
#         product_retriever = product_store.as_retriever(search_kwargs={"k": 3})




# ─── Utils ────────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


async def extract_text_from_file(uploaded: UploadFile) -> str:
    content = await uploaded.read()
    if uploaded.filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif uploaded.filename.lower().endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Only PDF or TXT allowed.")


# ─── Endpoint 1: Upload Product + Gold Example ────────────────────────────────
@app.post("/upload_product_features", response_model=UploadFeaturesResponse)
async def upload_product_features(
        product_file: UploadFile = File(...),
        gold_file: UploadFile = File(None)
):
    global product_store ,top_perf_store

    try:
        product_text = await extract_text_from_file(product_file)
        product_docs = chunk_text(product_text)
        product_store = FAISS.from_documents(product_docs, embedding=embeddings)
        product_store.save_local("db/product_docs_faiss")

        if gold_file:
            gold_text = await extract_text_from_file(gold_file)
            gold_docs = chunk_text(gold_text)
            top_perf_store = FAISS.from_documents(gold_docs, embedding=embeddings)
            top_perf_store.save_local("db/top_performers_faiss")

        return UploadFeaturesResponse(
            status="success",
            message="Product doc uploaded." + (" Gold example added." if gold_file else "")
        )
    except Exception as e:
        return UploadFeaturesResponse(
            status="error",
            message=f"Upload failed: {str(e)}"
        )


# ─── Endpoint 2: Evaluate Transcript ──────────────────────────────────────────
@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(data: EvaluateRequest):
    transcript = data.transcript

    top_examples = top_perf_store.similarity_search(transcript, k=3)
    top_text = "\n".join(doc.page_content for doc in top_examples)

    product_info = product_store.similarity_search(transcript, k=3)
    product_text = "\n".join(doc.page_content for doc in product_info)

    prompt = prompt_template.format(
        transcript=transcript,
        top_sample=top_text,
        product_info=product_text
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return EvaluateResponse(evaluation=response.content)


# ─── Endpoint 3: Evaluate Audio ───────────────────────────────────────────────
#
# @app.post("/evaluate_audio", response_model=EvaluateResponse)
# async def evaluate_audio(audio_file: UploadFile = File(...)):
#     try:
#         # Load audio file into memory buffer
#         audio_bytes = await audio_file.read()
#         audio_buffer = io.BytesIO(audio_bytes)
#
#         # Transcribe audio from buffer
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(audio_buffer) as source:
#             audio = recognizer.record(source)
#             transcript = recognizer.recognize_google(audio)
#
#         # Use same evaluation logic
#         top_examples = top_perf_retriever.get_relevant_documents(transcript)
#         top_text = "\n".join(doc.page_content for doc in top_examples)
#
#         product_info = product_retriever.get_relevant_documents(transcript)
#         product_text = "\n".join(doc.page_content for doc in product_info)
#
#         prompt = prompt_template.format(
#             transcript=transcript,
#             top_sample=top_text,
#             product_info=product_text
#         )
#         response = llm.invoke([{"role": "user", "content": prompt}])
#
#         return EvaluateResponse(evaluation=response.content)
#
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"evaluation": f"Audio Evaluation Failed: {str(e)}"}
#         )

# ─── Helper ───────────────────────────────────────────────────────────────────
def evaluate_transcript(transcript: str) -> str:
    # pull top examples
    top_examples = top_perf_store.similarity_search(transcript, k=3)
    top_text = "\n".join(doc.page_content for doc in top_examples)
    # pull product docs
    prod_examples = product_store.similarity_search(transcript, k=3)
    prod_text = "\n".join(doc.page_content for doc in prod_examples)
    # build prompt + call LLM
    prompt = prompt_template.format(
        transcript=transcript,
        top_sample=top_text,
        product_info=prod_text,
    )
    llm_resp = llm.invoke([{"role": "user", "content": prompt}])
    return llm_resp.content


# # Initialize real-time STT recorder
# recorder = AudioToTextRecorder(
#     spinner=False,
#     silero_sensitivity=0.01,
#     model="tiny.en",
#     language="en"  # <<< disable continuous mic capture

# )

# # Immediately turn its mic off so it won’t buffer live audio
# recorder.set_microphone(False)


# # ─── New /evaluate_audio Endpoint Using RealtimeSTT ──────────────────────────
# @app.post("/evaluate_audio_stt", response_model=EvaluateResponse)
# async def evaluate_audio(audio_file: UploadFile = File(...)):
#     try:
#         audio_bytes = await audio_file.read()
#         recorder.set_microphone(False)
#         recorder.feed_audio(audio_bytes)
#         transcript = recorder.text()
#         evaluation = evaluate_transcript(transcript)
#         print({"evaluation": evaluation, "transcript": transcript})
#         return EvaluateResponse(transcript=transcript, evaluation=evaluation)


#     except Exception as e:
#         return JSONResponse(status_code=500, content={"evaluation": str(e)})


# ─── New Whisper‐based Audio Endpoint ────────────────────────────────────────
@app.post("/evaluate_audio_whisper", response_model=EvaluateAudioResponse)
async def evaluate_audio(audio_file: UploadFile = File(...)):
    try:
        # 1) Read bytes
        audio_bytes = await audio_file.read()

        # 2) Write to a temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        model = whisper.load_model("tiny.en",device="cpu")  # or "base", "medium", "large"


        # 3) Run Whisper transcription
        whisper_result = model.transcribe(tmp_path)
        transcript = whisper_result["text"]

        # 4) Clean up temp file
        os.remove(tmp_path)

        # 5) Evaluate the transcript using your existing logic
        evaluation = evaluate_transcript(transcript)
        return EvaluateAudioResponse(transcript=transcript, evaluation=evaluation)

    except Exception as e:
        # 6) On error, return JSONResponse with the error
        return JSONResponse(
            status_code=500,
            content={"evaluation": f"Audio Evaluation Failed: {str(e)}"},
        )

# Mandatory Render port configuration
# At bottom of main.py
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=int(os.environ.get("PORT", 8000)),  # 8000 for local, $PORT in Render
#         workers=1
#     )
