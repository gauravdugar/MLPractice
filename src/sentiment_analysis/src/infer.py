import pickle

# Load model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_sentiment(input_text):
    prediction = model.predict([input_text])
    return prediction[0]

# Example usage
text = "I love this product! It's amazing."
print(predict_sentiment(text))
