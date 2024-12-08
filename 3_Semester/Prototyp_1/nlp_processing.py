import spacy

nlp = spacy.load("de_core_news_sm")

def analyze_text(text):
    """Analysiert den Text & extrahiert Entitäten & Sätze."""
    doc = nlp(text)
    return {
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "sentences": [sent.text for sent in doc.sents]
    }
