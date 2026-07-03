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
    page_title="CineAI — Movie Sentiment",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background: #0a0a0f;
    }

    /* Звёздное небо */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-image:
            radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 50% 10%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 90% 40%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 20% 90%, rgba(255,255,255,0.2) 0%, transparent 100%),
            radial-gradient(1px 1px at 80% 15%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(2px 2px at 60% 50%, rgba(255,215,0,0.2) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* Кинолента сверху */
    .film-strip {
        background: repeating-linear-gradient(
            90deg,
            #1a1a1a 0px, #1a1a1a 20px,
            #f5c518 20px, #f5c518 22px,
            #1a1a1a 22px, #1a1a1a 42px
        );
        height: 8px;
        width: 100%;
        margin-bottom: 2rem;
        opacity: 0.6;
    }

    .main-logo {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        color: #f5c518;
        letter-spacing: 8px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .main-title {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 40px rgba(245, 197, 24, 0.3);
        letter-spacing: -1px;
    }

    .main-title span {
        color: #f5c518;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
        letter-spacing: 1px;
        font-family: 'Inter', sans-serif;
    }

    /* Карточки фильмов */
    .movies-section {
        margin: 1.5rem 0;
    }

    .movies-label {
        color: #6b7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        font-family: 'Inter', sans-serif;
    }

    .movies-grid {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }

    .movie-chip {
        background: rgba(245, 197, 24, 0.08);
        border: 1px solid rgba(245, 197, 24, 0.2);
        border-radius: 50px;
        padding: 0.4rem 0.9rem;
        font-size: 0.8rem;
        color: #f5c518;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .movie-chip:hover {
        background: rgba(245, 197, 24, 0.15);
        border-color: rgba(245, 197, 24, 0.5);
    }

    /* Поле ввода */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(245, 197, 24, 0.2) !important;
        border-radius: 16px !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(245, 197, 24, 0.6) !important;
        box-shadow: 0 0 25px rgba(245, 197, 24, 0.08) !important;
    }

    .stTextArea label {
        color: #9ca3af !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* Кнопка */
    .stButton button {
        background: linear-gradient(135deg, #f5c518, #e6a800) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
        padding: 0.7rem !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(245, 197, 24, 0.3) !important;
    }

    /* Результат */
    .result-positive {
        background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(16,185,129,0.03));
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .result-negative {
        background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(239,68,68,0.03));
        border: 1px solid rgba(248,113,113,0.3);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .result-emoji { font-size: 4.5rem; margin-bottom: 0.8rem; }

    .result-label-pos {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #34d399;
        margin-bottom: 0.3rem;
    }

    .result-label-neg {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #f87171;
        margin-bottom: 0.3rem;
    }

    .result-desc {
        color: #6b7280;
        font-size: 0.85rem;
        font-family: 'Inter', sans-serif;
        margin-bottom: 1.2rem;
    }

    .stars-positive { color: #f5c518; font-size: 1.5rem; letter-spacing: 4px; }
    .stars-negative { color: #374151; font-size: 1.5rem; letter-spacing: 4px; }

    .confidence-wrap {
        margin: 1rem auto;
        max-width: 280px;
    }

    .confidence-label {
        color: #6b7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }

    .bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 50px;
        height: 6px;
        overflow: hidden;
    }

    .bar-pos {
        background: linear-gradient(90deg, #34d399, #059669);
        height: 100%;
        border-radius: 50px;
    }

    .bar-neg {
        background: linear-gradient(90deg, #f87171, #dc2626);
        height: 100%;
        border-radius: 50px;
    }

    .confidence-pct {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        margin-top: 0.5rem;
    }

    .confidence-pct-pos { color: #34d399; }
    .confidence-pct-neg { color: #f87171; }

    /* Статистика */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }

    .stat-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 100px;
    }

    .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #f5c518;
    }

    .stat-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 2rem 0;
    }

    .footer {
        text-align: center;
        color: #1f2937;
        font-size: 0.75rem;
        font-family: 'Inter', sans-serif;
        padding-bottom: 2rem;
        letter-spacing: 0.5px;
    }

    .imdb-badge {
        display: inline-block;
        background: #f5c518;
        color: #000;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 1px;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Кинолента
st.markdown('<div class="film-strip"></div>', unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-logo">✦ &nbsp; CineAI &nbsp; ✦</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Movie <span>Sentiment</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered · Trained on 50,000 IMDB reviews · Instant analysis</div>', unsafe_allow_html=True)

# Статистика
st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-num">89.5%</div>
        <div class="stat-desc">Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">50K</div>
        <div class="stat-desc">Reviews</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">10K</div>
        <div class="stat-desc">Features</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">&lt;1s</div>
        <div class="stat-desc">Response</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Примеры фильмов
st.markdown("""
<div class="movies-section">
    <div class="movies-label">🎬 Try a review about</div>
    <div class="movies-grid">
        <div class="movie-chip">⭐ The Shawshank Redemption</div>
        <div class="movie-chip">🌀 Inception</div>
        <div class="movie-chip">🦇 The Dark Knight</div>
        class="movie-chip">👾 Interstellar</div>
        <div class="movie-chip">🎭 Parasite</div>
        <div class="movie-chip">🔫 Pulp Fiction</div>
        <div class="movie-chip">👁 Schindler's List</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Поле ввода
review = st.text_area(
    "Your review",
    height=180,
    placeholder="e.g. A masterpiece of modern cinema. The direction, the acting, the score — everything was absolutely flawless..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🎬 Analyze", use_container_width=True)

# Предсказание
if analyze:
    if review.strip() == '':
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Reading between the lines..."):
            cleaned = clean_text(review)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            confidence = probability[1] * 100
            stars = "★★★★★" if confidence > 80 else "★★★★☆"
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-emoji">🌟</div>
                <div class="result-label-pos">Standing Ovation</div>
                <div class="result-desc">The audience loved it</div>
                <div class="stars-positive">{stars}</div>
                <div class="confidence-wrap">
                    <div class="confidence-label">Confidence</div>
                    <div class="bar-bg">
                        <div class="bar-pos" style="width:{confidence}%"></div>
                    </div>
                    <div class="confidence-pct confidence-pct-pos">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = probability[0] * 100
            stars = "★☆☆☆☆" if confidence > 80 else "★★☆☆☆"
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-emoji">🎭</div>
                <div class="result-label-neg">Walk Out</div>
                <div class="result-desc">The audience wasn't impressed</div>
                <div class="stars-negative">{stars}</div>
                <div class="confidence-wrap">
                    <div class="confidence-label">Confidence</div>
                    <div class="bar-bg">
                        <div class="bar-neg" style="width:{confidence}%"></div>
                    </div>
                    <div class="confidence-pct confidence-pct-neg">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <span class="imdb-badge">IMDB</span> &nbsp;
    Trained on 50,000 reviews · Built with Python & scikit-learn · English only
</div>
""", unsafe_allow_html=True)
