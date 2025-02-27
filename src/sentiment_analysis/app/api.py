from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
with open('../models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict/")
async def predict_sentiment(text: str):
    prediction = model.predict([text])
    return {"sentiment": prediction[0]}

# Run API
# uvicorn app.api:app --reload
