import streamlit as st
from streamlit_audiorec import st_audiorec
from groq_layer import analyze_sentiment, extract_rating
import whisper
import tempfile
import os
import time
import pandas as pd

# Load Whisper model
whisper_model = whisper.load_model("base")

# Page setup
st.set_page_config(page_title="Sentiment Studio", page_icon="🎧", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🎧 Sentiment Studio")
    st.markdown("Analyze **text**, **voice recordings**, or **uploaded audio files** for sentiment, tone, and key themes.")
    st.markdown("Powered by Whisper + LLaMA-3 via Groq.")
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit, Whisper, and Groq.")

st.title("📊 Text & Voice Sentiment Analyzer")

def render_sentiment_ui(summary: str, rating: int):
    if rating >= 4:
        delta = "Positive"
        delta_color = "normal"
    elif rating == 3:
        delta = "Neutral"
        delta_color = "off"
    else:
        delta = "Negative"
        delta_color = "inverse"

    st.markdown("### 💬 Sentiment Analysis Result")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric(label="Rating", value=f"{rating}/5", delta=delta, delta_color=delta_color)
    with col2:
        st.write(summary)

# === VOICE RECORDING SECTION ===
st.header("🎙️ Record Live Audio")

audio_bytes = st_audiorec()

if audio_bytes:
    st.success("✅ Recording captured!")
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name

    st.info("Transcribing with Whisper...")
    result = whisper_model.transcribe(audio_path)
    transcribed_text = result["text"]
    os.remove(audio_path)

    st.markdown("**📝 Transcribed Text:**")
    st.code(transcribed_text, language="markdown")

    st.info("Analyzing sentiment with LLaMA-3...")
    summary = analyze_sentiment(transcribed_text)
    rating = extract_rating(summary)

    render_sentiment_ui(summary, rating)

st.markdown("---")

# === MULTI AUDIO FILE UPLOAD ===
st.header("📂 Upload Audio Files")
uploaded_audio_files = st.file_uploader(
    "Upload `.wav`, `.mp3`, or `.m4a` files", 
    accept_multiple_files=True, 
    type=["wav", "mp3", "m4a"]
)

audio_results = []
all_transcripts = []

if uploaded_audio_files:
    for audio_file in uploaded_audio_files:
        st.subheader(f"🎵 {audio_file.name}")
        st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_audio_path = tmp_file.name

        st.info("Transcribing with Whisper...")
        result = whisper_model.transcribe(tmp_audio_path)
        transcript = result["text"]
        all_transcripts.append(transcript)

        st.markdown("**📝 Transcribed Text:**")
        st.code(transcript, language="markdown")

        st.info("Analyzing sentiment...")
        summary = analyze_sentiment(transcript)
        rating = extract_rating(summary)

        render_sentiment_ui(summary, rating)

        audio_results.append({
            "File": audio_file.name,
            "Transcript": transcript,
            "Summary": summary,
            "Rating": rating
        })

        os.remove(tmp_audio_path)

    st.markdown("---")
    st.subheader("📊 Overall Summary (All Uploaded Audio Files)")
    combined_transcripts = "\n\n".join(all_transcripts)
    overall_summary = analyze_sentiment(combined_transcripts)
    overall_rating = extract_rating(overall_summary)

    render_sentiment_ui(overall_summary, overall_rating)

    if st.button("📤 Export Audio Analysis to CSV"):
        df = pd.DataFrame(audio_results)
        csv_path = "/tmp/audio_sentiment.csv"
        df.to_csv(csv_path, index=False)
        st.download_button("Download CSV", data=open(csv_path, "rb"), file_name="audio_sentiment.csv", mime="text/csv")

st.markdown("---")

# === TEXT FILE UPLOAD SECTION ===
st.header("📄 Upload Text Files for Sentiment Analysis")
uploaded_files = st.file_uploader("Upload `.txt` files", accept_multiple_files=True, type=["txt"])

text_results = []

if uploaded_files:
    all_texts = []
    file_summaries = []
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        content = uploaded_file.read().decode("utf-8")
        all_texts.append(content)

        summary = analyze_sentiment(content)
        rating = extract_rating(summary)
        file_summaries.append((uploaded_file.name, summary, rating))
        text_results.append({
            "File": uploaded_file.name,
            "Text": content,
            "Summary": summary,
            "Rating": rating
        })

        progress_bar.progress((idx + 1) / len(uploaded_files))
        time.sleep(0.1)

    combined_text = "\n\n".join(all_texts)
    overall_summary = analyze_sentiment(combined_text)
    overall_rating = extract_rating(overall_summary)

    st.markdown("---")
    st.subheader("📊 Overall Summary (All Text Files)")
    render_sentiment_ui(overall_summary, overall_rating)

    st.markdown("---")
    st.subheader("📁 Individual File Summaries")

    for filename, summary, rating in file_summaries:
        with st.expander(f"📄 {filename} — Rating: {rating}/5"):
            st.write(summary)

    if st.button("📤 Export Text Analysis to CSV"):
        df = pd.DataFrame(text_results)
        csv_path = "/tmp/text_sentiment.csv"
        df.to_csv(csv_path, index=False)
        st.download_button("Download CSV", data=open(csv_path, "rb"), file_name="text_sentiment.csv", mime="text/csv")
else:
    st.info("📥 Upload one or more `.txt` files above to analyze written content.")
