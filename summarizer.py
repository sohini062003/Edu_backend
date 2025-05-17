import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
 
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tfidf_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    scores = sim_matrix.sum(axis=1)
    top_n_idx = np.argsort(scores)[-num_sentences:]
    selected_sentences = [sentences[i] for i in sorted(top_n_idx)]
    summary = ' '.join(selected_sentences)
    return summary, selected_sentences

def llm_summary(user_input, style, n):
    prompt = [
        SystemMessage(content=f"""
        You are Summarizer, an AI assistant specialized in academic summarization for students.
        Your job is to summarize any input text into a clear, structured, and useful format for students.
        Follow the style specified by the user and generate {n} key points.
        Style options include:
        
        bullet: generate {n} clear, concise bullet points.
        qa: generate {n} question-answer pairs based on the input.
        paragraph: generate a coherent paragraph with approximately {n} lines.
        
        Strictly follow the style. Do not include extra comments.
        Current style: {style}
        """),
        HumanMessage(content=user_input)
    ]

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,        
        temperature=0.5
    )

    res = llm.invoke(prompt)
    print("🔎 Raw LLM Response:", res.content) 
    return res.content

