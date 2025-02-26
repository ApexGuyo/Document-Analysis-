import streamlit as st
import fitz  # PyMuPDF
import docx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import openai
import os
import langdetect
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from openai import OpenAI

# Set OpenAI API Key (replace with your key)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Load Spacy NLP model
nlp_spacy = spacy.load("en_core_web_sm")

# Load Hugging Face summarization model
bart_model = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

client = OpenAI()


def load_document(file):
    """Extracts text from a document (PDF, DOCX, TXT)."""
    text = ""
    if file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text("text")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:  # Plain text file
        text = file.read().decode("utf-8")
    return text


def detect_language(text):
    """Detects the language of the text."""
    return langdetect.detect(text)


def summarize_text(text, lang):
    """Summarizes the document using OpenAI's GPT if available, otherwise falls back to Hugging Face."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change from "gpt-4"
            messages=[
                {"role": "user", "content": f"Summarize this in {lang}: {text[:2000]}"}]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return "Summarization failed. Please try again."


def extract_keywords(text):
    """Extracts key phrases using Hugging Face transformers."""
    nlp = pipeline(
        "ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = nlp(text[:1000])
    keywords = list(set([ent["word"] for ent in entities]))
    return keywords


def analyze_sentiment(text):
    """Performs sentiment analysis using Hugging Face."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text[:1000])
    return result[0]


def generate_wordcloud(text):
    """Generates a word cloud visualization from the document text."""
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


def plot_sentiment(sentiment):
    """Visualizes sentiment score."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 3))
    sns.barplot(x=[sentiment["label"]], y=[sentiment["score"]], palette=[
                "green" if sentiment["label"] == "POSITIVE" else "red"])
    plt.ylim(0, 1)
    plt.title("Sentiment Score")
    st.pyplot(plt)


def compute_readability(text):
    """Computes the readability score of the text using the Flesch-Kincaid score."""
    words = text.split()
    sentences = text.count('.')
    syllables = sum(len(word) for word in words) / len(words)
    if len(words) == 0 or sentences == 0:
        return "N/A"
    score = 206.835 - 1.015 * (len(words) / sentences) - \
        84.6 * (syllables / len(words))
    return round(score, 2)


def extract_named_entities(text):
    """Extracts named entities using spaCy."""
    doc = nlp_spacy(text[:1000])
    entities = {ent.label_: set() for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].add(ent.text)
    return entities


def display_named_entities(entities):
    """Displays named entities in a structured format."""
    for entity_type, entity_list in entities.items():
        st.subheader(f"🔹 {entity_type}")
        st.write(", ".join(entity_list))


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

        # Language Detection
        language = detect_language(doc_text)
        st.subheader("🌍 Detected Language")
        st.write(f"Language: {language}")

        # Summarization
        st.subheader("📌 Summary")
        summary = summarize_text(doc_text, language)
        st.write(summary)

        # Keywords
        st.subheader("🔑 Key Topics")
        keywords = extract_keywords(doc_text)
        st.write(", ".join(keywords))

        # Sentiment Analysis
        st.subheader("📊 Sentiment Analysis")
        sentiment = analyze_sentiment(doc_text)
        st.write(
            f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
        plot_sentiment(sentiment)

        # Word Cloud
        st.subheader("☁ Word Cloud Visualization")
        generate_wordcloud(doc_text)

        # Readability Score
        st.subheader("📖 Readability Score")
        readability = compute_readability(doc_text)
        st.write(f"Flesch-Kincaid Score: {readability}")

        # Named Entity Recognition
        st.subheader("🔍 Named Entities")
        named_entities = extract_named_entities(doc_text)
        display_named_entities(named_entities)

