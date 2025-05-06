import streamlit as st
import requests
from io import BytesIO
from streamlit_mic_recorder import mic_recorder

BASE_URL = "http://127.0.0.1:8000"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG-Eval by Venkat Iyer", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Sidebar customization */
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: #ecf0f1;
        }
        [data-testid="stSidebar"] .css-1d391kg {
            color: #ecf0f1;
        }
        /* Main content area */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1.2rem;
            border: none;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #388e3c;
        }
        textarea, input[type="file"] {
            border-radius: 8px !important;
        }
        /* Radio button labels */
        label[data-testid="stRadioLabel"] > div {
            font-size: 1.1rem;
            font-weight: 500;
            color: #ecf0f1;
        }
        div[role="radiogroup"] > label:hover {
            background-color: #34495e;
            border-radius: 0.5rem;
        }
        /* Selected radio button styling */
        input[type="radio"]:checked + div {
            background-color: #364e41 !important; /* Charcoal green */
            color: white !important;
            border-radius: 0.5rem;
        }
        /* Sidebar navigation items with logos */
        .sidebar-nav-item {
            display: flex;
            align-items: center;
            padding: 8px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
        }
        .sidebar-nav-item img {
            width: 24px;
            height: 24px;
            margin-right: 8px;
        }
        .sidebar-nav-item:hover {
            background-color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ───────────────────────────────────────────────────────


# Inject custom CSS
st.markdown("""
    <style>
    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        # color: #4d2ba5;
        margin-bottom: 15px;
         margin-top: -40px;80
         # background: linear-gradient(90deg, #0021F3, #9400D3, #EE82EE);
    background: linear-gradient(120deg, #9400D3, #5da6b0, #134f5c);



    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
        # text-shadow: 2px 2px 4px black;

    }
    .sidebar-step {
        font-size: 18px;
        color: #999999;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Use HTML with CSS classes in the sidebar
st.sidebar.markdown('<div class="sidebar-title">RAG evaluation made effortless ! </div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-step">1. Upload doc & examples</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-step">2. Input (text/voice)</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-step">3. Get score</div>', unsafe_allow_html=True)




st.sidebar.markdown("")


import streamlit as st

# Custom CSS for styling the radio button in the sidebar
st.markdown("""
    <style>
        /* Targeting the radio button label */
        div[data-testid="stSidebar"] div[role="radiogroup"] div[aria-labelledby] {
            font-family: 'Arial', sans-serif !important; /* Change font family */
            font-size: 16px !important; /* Change font size */
            font-weight: bold !important; /* Make text bold */
        }

        /* Adding an icon before each radio option */
        div[data-testid="stSidebar"] div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] label:before {
            content: '🔹'; /* Unicode character for a bullet point */
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar radio button widget
page = st.sidebar.radio("*Go to*", ["Upload", "Transcript Evaluation", "Audio Evaluation"])



# Inject CSS to enlarge radio labels and sidebar text
st.markdown("""
    <style>
      /* 1) Radio‐option text lives in a <span> inside each label */
      div[role="radiogroup"] > label > span {
        font-size: 1.8rem !important;    /* bump this to taste */
        font-weight: 600  !important;    /* make it bold */
      }
      /* 2) Any plain Markdown paragraph in the sidebar */
      div[data-testid="stSidebar"] p {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
      }
      /* 3) Increase the gap between radio options (you already saw this work) */
      div[role="radiogroup"] {
        gap: 1.5rem !important;
      }
    </style>
""", unsafe_allow_html=True)

# # Set page title and favicon
# st.set_page_config(
#     page_title="Rag Eval",
#     layout="wide"
# )

# 2) Render icon + header side by side using columns
col_icon, col_title = st.columns([0.075, 1], gap="small")
with col_icon:
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.image("static/recognition.png", width=50)        # your 40px-wide icon
with col_title:
    st.markdown("## RAG-Eval")                   # H2, so size matches a big page header


# ── Gradient Header with Separate Gradients ──────────────────────────────────
st.markdown("""
<h2 style="margin-top:20px; font-weight:bold;">
  <span style="
    background: linear-gradient(90deg, #0021F3, #9400D3, #EE82EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  ">Hello there!</span>
  <span style="
    background: linear-gradient(90deg, #8A2BE2, #FF69B4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  "> Ready to evaluate ?</span>
</h2>
""", unsafe_allow_html=True)



# ─── Page 1: Upload ───────────────────────────────────────────────────────────
if page == "Upload":
    st.image("static/upload.png", width=50)  # Display upload section logo
    with st.expander("📁 Upload Product Features and Gold Example", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            product_file = st.file_uploader("Upload Product Features (PDF/TXT)", type=["pdf", "txt"])
        with col2:
            gold_file = st.file_uploader("Upload Gold Example (PDF/TXT)", type=["pdf", "txt"])

        if st.button("⬆️ Upload Files"):
            if product_file:
                files = {"product_file": product_file}
                if gold_file:
                    files["gold_file"] = gold_file
                try:
                    with st.spinner("📤 Uploading files..."):
                        resp = requests.post(f"{BASE_URL}/upload_product_features", files=files)
                    if resp.ok:
                        st.success(resp.json().get("message", "✅ Upload successful."))
                    else:
                        st.error(f"❌ Error uploading files: {resp.status_code}")
                except Exception as e:
                    st.error(f"🚫 Failed to connect to backend: {e}")
            else:
                st.warning("⚠️ Please upload the product file.")

# ─── Page 2: Transcript Evaluation ────────────────────────────────────────────
elif page == "Transcript Evaluation":
    st.image("static/transcript.png", width=50)  # Display transcript evaluation logo
    with st.expander("📝 Enter and Evaluate Transcript", expanded=True):
        txt = st.text_area("Enter Transcript for Evaluation")

        if st.button("🔍 Evaluate Transcript"):
            if txt:
                try:
                    with st.spinner("🔎 Evaluating transcript..."):
                        resp = requests.post(f"{BASE_URL}/evaluate", json={"transcript": txt})
                    if resp.ok:
                        st.subheader("📊 Evaluation Results")
                        st.success(resp.json()["evaluation"])
                    else:
                        st.error(f"❌ Error evaluating transcript: {resp.status_code}")
                except Exception as e:
                    st.error(f"🚫 Failed to connect to backend: {e}")
            else:
                st.warning("⚠️ Please enter a transcript.")

# ─── Page 3: Audio Evaluation ─────────────────────────────────────────────────
elif page == "Audio Evaluation":
    st.image("static/mic.png", width=50)  # Display audio evaluation logo
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
            wav_bytes = st.session_state["wav_bytes"]
            st.audio(wav_bytes, format="audio/wav")

            if st.button("🎧 Evaluate Audio", key="eval_audio"):
                try:
                    files = {
                        "audio_file": (
                            "recorded.wav",
                            BytesIO(wav_bytes),
                            "audio/wav"
                        )
                    }
                    with st.spinner("🔎 Evaluating audio..."):
                        resp = requests.post(f"{BASE_URL}/evaluate_audio_whisper", files=files)
                    if resp.ok:
                        result = resp.json()
                        st.subheader("🗒️ Transcript:")
                        st.code(result.get("transcript", "—"))
                        st.subheader("📊 Evaluation:")
                        st.success(result.get("evaluation", "—"))
                    else:
                        st.error(f"❌ Error: {resp.status_code} {resp.text}")
                except Exception as e:
                    st.error(f"🚫 Request failed: {e}")

            # if st.button("🔁 Clear Recording", key="clear_audio"):
            #     del st.session_state["wav_bytes"]
            #     st.experimental_rerun()
