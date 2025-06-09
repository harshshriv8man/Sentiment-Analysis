from groq import Groq
import streamlit as st
import re


api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

SENTIMENT_SYSTEM = """
You are a highly skilled sentiment analysis assistant. You will analyze transcribed voice or written text from users. 

Your responsibilities:
- Accurately summarize the overall sentiment: Positive, Negative, Neutral, or Mixed.
- Assign a confidence-based rating from 1 (very negative) to 5 (very positive).
- Go deeper than just tone: analyze intent, emotional cues, frustration, joy, sarcasm, or ambiguity.
- Identify recurring themes or concerns. Highlight key words or emotional triggers.
- Be detailed and nuanced, especially for spoken content. Reflect vocal indicators (if mentioned or implied).
- Avoid superficial judgments. Do not default to Neutral unless truly balanced.
- Format your response clearly:

Example format:
Sentiment: Positive
Rating: 4/5
Themes: fast service, friendly staff, good experience
Summary: The user expresses overall satisfaction, especially with service speed and friendliness.

Respond in clear, structured plain text. No unnecessary repetition of the original text. Prioritize clarity and depth.
"""

def analyze_sentiment(text: str) -> str:
    messages = [
        {"role": "system", "content": SENTIMENT_SYSTEM},
        {"role": "user", "content": f"User text:\n{text}"}
    ]
    res = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return res.choices[0].message.content.strip()

def extract_rating(text: str) -> int:
    match = re.search(r"(rating|score)[^\d]*(\d)", text, re.IGNORECASE)
    return int(match.group(2)) if match else 3
