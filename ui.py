import streamlit as st
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
from logic import (
    upload_product_features,
    evaluate_transcript,
    evaluate_audio_whisper,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG-Eval by Venkat Iyer", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Sidebar customization */
        [data-testid="stSidebar"] { background-color: #2c3e50; color: #ecf0f1; }
        [data-testid="stSidebar"] .css-1d391kg { color: #ecf0f1; }
        /* Main content area */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #2c3e50; }
        .stButton > button {
            background-color: #4CAF50; color: white; padding: 0.5rem 1.2rem;
            border: none; border-radius: 8px; font-size: 16px;
        }
        .stButton > button:hover { background-color: #388e3c; }
        textarea, input[type="file"] { border-radius: 8px !important; }
        /* Radio button labels */
        label[data-testid="stRadioLabel"] > div {
            font-size: 1.1rem; font-weight: 500; color: #ecf0f1;
        }
        div[role="radiogroup"] > label:hover { background-color: #34495e; border-radius: 0.5rem; }
        input[type="radio"]:checked + div {
            background-color: #364e41 !important; color: white !important;
            border-radius: 0.5rem;
        }
        .sidebar-nav-item { display: flex; align-items: center;
            padding: 8px 16px; border-radius: 8px; margin-bottom: 8px; cursor: pointer;
        }
        .sidebar-nav-item img { width: 24px; height: 24px; margin-right: 8px; }
        .sidebar-nav-item:hover { background-color: #34495e; }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 28px; font-weight: bold; margin-bottom: 15px;
        background: linear-gradient(120deg, #9400D3, #5da6b0, #134f5c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sidebar-step { font-size: 18px; color: #999999; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">RAG evaluation made effortless !</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-step">1. Upload doc & examples</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-step">2. Input (text/voice)</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-step">3. Get score</div>', unsafe_allow_html=True)

st.markdown("""
<h2 style="margin-top:20px; font-weight:bold;">
  <span style="background: linear-gradient(90deg, #0021F3, #9400D3, #EE82EE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    Hello there!</span>
  <span style="background: linear-gradient(90deg, #8A2BE2, #FF69B4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    Ready to evaluate?</span>
</h2>
""", unsafe_allow_html=True)

# ─── Sidebar Page Navigation ─────────────────────────────────────────────────
page = st.sidebar.radio("*Go to*", ["Upload", "Transcript Evaluation", "Audio Evaluation"])

# Header
col_icon, col_title = st.columns([0.075, 1], gap="small")
with col_icon:
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.image("static/recognition.png", width=50)
with col_title:
    st.markdown("## RAG-Eval")

# ─── Helper Wrappers ──────────────────────────────────────────────────────────
class _UploadWrapper:
    def __init__(self, st_file):
        self._file = st_file
        self.filename = st_file.name

    def read(self):
        return self._file.getvalue()

class _BytesWrapper:
    def __init__(self, b, name="audio.wav"):
        self._bytes = b
        self.filename = name

    async def read(self):
        return self._bytes

# ─── Page 1: Upload ───────────────────────────────────────────────────────────
if page == "Upload":
    st.image("static/upload.png", width=50)
    with st.expander("📁 Upload Product Features and Gold Example", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            product_file = st.file_uploader("Upload Product Features (PDF/TXT)", type=["pdf", "txt"])
        with col2:
            gold_file = st.file_uploader("Upload Gold Example (PDF/TXT)", type=["pdf", "txt"])

        if st.button("⬆️ Upload Files"):
            if not product_file:
                st.warning("⚠️ Please upload the product file.")
            else:
                wrapped_prod = _UploadWrapper(product_file)
                wrapped_gold = _UploadWrapper(gold_file) if gold_file else None

                with st.spinner("📤 Processing upload..."):
                    result = upload_product_features(wrapped_prod, wrapped_gold)
                if result["status"] == "success":
                    st.success(result["message"])
                else:
                    st.error(result["message"])

# ─── Page 2: Transcript Evaluation ────────────────────────────────────────────
elif page == "Transcript Evaluation":
    st.image("static/transcript.png", width=50)
    with st.expander("📝 Enter and Evaluate Transcript", expanded=True):
        txt = st.text_area("Enter Transcript for Evaluation")
        if st.button("🔍 Evaluate Transcript"):
            if not txt.strip():
                st.warning("⚠️ Please enter a transcript.")
            else:
                result = evaluate_transcript(txt)
                st.subheader("📊 Evaluation Results")
                st.markdown(result, unsafe_allow_html=True)

# ─── Page 3: Audio Evaluation ─────────────────────────────────────────────────
elif page == "Audio Evaluation":
    st.image("static/mic.png", width=50)
    with st.expander("🎙️ Record and Evaluate Audio", expanded=True):
        audio_data = mic_recorder(
            start_prompt="🎤 Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            format="wav",
            key="wave_recorder"
        )

        if audio_data and "wav_bytes" not in st.session_state:
            st.session_state["wav_bytes"] = audio_data["bytes"]

        if "wav_bytes" in st.session_state:
            wav = st.session_state["wav_bytes"]
            st.audio(wav, format="audio/wav")

            if st.button("🎧 Evaluate Audio"):
                wrapped_audio = _BytesWrapper(wav)
                with st.spinner("🔎 Evaluating audio with Whisper..."):
                    result = evaluate_audio_whisper(wrapped_audio)

                st.subheader("🗒️ Transcript:")
                st.code(result.get("transcript", "—"))
                st.subheader("📊 Evaluation:")
                st.markdown(result.get("evaluation", "—"), unsafe_allow_html=True)

