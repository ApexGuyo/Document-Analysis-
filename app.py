import streamlit as st
import PyMuPDF
import docx
from transformers import pipeline
import openai
import os

# Set OpenAI API Key (replace with your key)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


def load_document(file):
    """Extracts text from a document (PDF, DOCX, TXT)."""
    text = ""
    if file.type == "application/pdf":
        pdf = PyMuPDF.open(file)
        for page in pdf:
            text += page.get_text("text")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:  # Plain text file
        text = file.read().decode("utf-8")
    return text


def summarize_text(text):
    """Summarizes the document using OpenAI's GPT."""
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": f"Summarize this: {text[:2000]}"}]
    )
    return response["choices"][0]["message"]["content"]


def extract_keywords(text):
    """Extracts key phrases using Hugging Face transformers."""
    nlp = pipeline(
        "ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = nlp(text[:1000])
    keywords = list(set([ent["word"] for ent in entities]))
    return ", ".join(keywords)


def analyze_sentiment(text):
    """Performs sentiment analysis using Hugging Face."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text[:1000])
    return result[0]


# Streamlit UI
st.title("📄 LLM Document Analyzer")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    st.write("Processing document...")
    doc_text = load_document(uploaded_file)

    if doc_text:
        st.subheader("📜 Extracted Text")
        st.text_area("Document Content", doc_text[:1000] + "...", height=200)

        # Summarization
        st.subheader("📌 Summary")
        summary = summarize_text(doc_text)
        st.write(summary)

        # Keywords
        st.subheader("🔑 Key Topics")
        keywords = extract_keywords(doc_text)
        st.write(keywords)

        # Sentiment Analysis
        st.subheader("📊 Sentiment Analysis")
        sentiment = analyze_sentiment(doc_text)
        st.write(
            f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
