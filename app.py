import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
        letter-spacing: 0.5px;
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 16px !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid rgba(167, 139, 250, 0.8) !important;
        box-shadow: 0 0 20px rgba(167, 139, 250, 0.15) !important;
    }

    .stButton button {
        background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }

    .result-positive {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-negative {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.1), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-positive .result-label { color: #34d399; }
    .result-negative .result-label { color: #f87171; }

    .confidence-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        height: 8px;
        margin: 1rem auto;
        max-width: 300px;
        overflow: hidden;
    }

    .confidence-bar-positive {
        background: linear-gradient(90deg, #34d399, #059669);
        height: 100%;
        border-radius: 50px;
        transition: width 0.8s ease;
    }

    .confidence-bar-negative {
        background: linear-gradient(90deg, #f87171, #dc2626);
        height: 100%;
        border-radius: 50px;
        transition: width 0.8s ease;
    }

    .confidence-text {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2.5rem;
        flex-wrap: wrap;
    }

    .stat-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .footer {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-bottom: 2rem;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by TF-IDF & Logistic Regression · Trained on 50K IMDB reviews</div>', unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-value">89.5%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">50K</div>
        <div class="stat-label">Reviews trained</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">10K</div>
        <div class="stat-label">TF-IDF features</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Input
review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="e.g. This movie was absolutely brilliant! The performances were outstanding and the story kept me on the edge of my seat..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("✨ Analyze Sentiment", use_container_width=True)

# Prediction
if analyze:
    if review.strip() == '':
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(review)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            confidence = probability[1] * 100
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-emoji">🌟</div>
                <div class="result-label">Positive Review</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-positive" style="width: {confidence}%"></div>
                </div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = probability[0] * 100
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-emoji">💔</div>
                <div class="result-label">Negative Review</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-negative" style="width: {confidence}%"></div>
                </div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with Python · scikit-learn · Streamlit &nbsp;·&nbsp; English reviews only
</div>
""", unsafe_allow_html=True)
