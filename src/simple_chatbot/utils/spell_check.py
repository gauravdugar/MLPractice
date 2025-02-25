from spellchecker import SpellChecker

spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word)
    return " ".join(corrected_words)
