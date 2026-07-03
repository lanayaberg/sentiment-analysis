import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
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

st.title("🎬 Анализ тональности отзывов")
st.write("Введи отзыв на фильм — модель определит позитивный он или негативный")

review = st.text_area("Твой отзыв:", height=150, placeholder="This movie was absolutely amazing...")

if st.button("Определить тональность"):
    if review.strip() == '':
        st.warning("Введи текст отзыва!")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ Позитивный отзыв (уверенность: {probability[1]*100:.1f}%)")
        else:
            st.error(f"❌ Негативный отзыв (уверенность: {probability[0]*100:.1f}%)")