import spacy

nlp = spacy.load('en_core_web_sm')

def analyze_text(text):
    doc = nlp(text)
    analysis = [(token.text, token.pos_, token.dep_) for token in doc]
    analysis_str = ", ".join(
        [f"Token: {token_text}, POS: {pos}, Dependency: {dep}" for token_text, pos, dep in analysis]
    )
    print(f"Analysis: [{analysis_str}]")
    return analysis