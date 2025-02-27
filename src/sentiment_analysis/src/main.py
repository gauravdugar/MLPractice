import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --------- Preprocessing Function ---------
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# --------- Load Dataset ---------
print("Loading dataset...")
train_df = pd.read_csv('../data/imdb_train.csv')
test_df = pd.read_csv('../data/imdb_test.csv')

# Preprocess text data
print("Preprocessing text data...")
train_df['clean_text'] = train_df['text'].astype(str).apply(clean_text)
test_df['clean_text'] = test_df['text'].astype(str).apply(clean_text)

X_train, y_train = train_df['clean_text'], train_df['label']
X_test, y_test = test_df['clean_text'], test_df['label']

# --------- Train Model ---------
print("Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to numerical features
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

pipeline.fit(X_train, y_train)

# Evaluate model
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
model_path = '../models/sentiment_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"Model saved at {model_path}")

# --------- Inference Function ---------
def predict_sentiment(text):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict([text])
    return prediction[0]

# Example usage
sample_review = "One time must watch."
print(f"Sample Prediction: {predict_sentiment(sample_review)}")
