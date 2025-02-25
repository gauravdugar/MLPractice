from fuzzywuzzy import process

known_phrases = {
    "what's your name": "My name is SimpleChatbot.",
    # Add other key phrases and their responses
}

def match_intent(query):
    best_match, score = process.extractOne(query, list(known_phrases.keys()))
    if score >= 80:
        return known_phrases[best_match]
    return None