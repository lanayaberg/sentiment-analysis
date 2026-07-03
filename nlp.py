import pandas as pd

df = pd.read_csv('data/IMDB Dataset.csv')

print(df.shape)
print(df.head())
print(df['sentiment'].value_counts())
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv('data/IMDB Dataset.csv')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Убираем HTML теги
    text = re.sub(r'<.*?>', '', text)
    # Убираем всё кроме букв
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # В нижний регистр
    text = text.lower()
    # Убираем стоп-слова (the, a, is, in — не несут смысла)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)

print(df['clean_review'].head(3))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# WordCloud для позитивных отзывов
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_review'])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Самые частые слова в позитивных отзывах')
plt.savefig('images/wordcloud_positive.png')
plt.close()

# WordCloud для негативных отзывов
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_review'])
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Самые частые слова в негативных отзывах')
plt.savefig('images/wordcloud_negative.png')
plt.close()

print("WordCloud сохранены!")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Превращаем текст в числа через TF-IDF
tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оцениваем
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Сохраняем модель и векторайзер
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("Модель сохранена!")