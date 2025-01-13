import spacy


try:
    nlp = spacy.load("de_core_news_lg")
except Exception as e:
    raise ImportError("Modell konnte nicht geladen werden: {e}")

def analyze_text(text):
    """
    Analysiert den bereinigten Text und extrahiert Entitäten und Sätze.
    """
    doc = nlp(text)

    # Entferne Duplikate bei Entitäten
    unique_entities = list(set((ent.text, ent.label_) for ent in doc.ents))

    return {
        "entities": unique_entities,
        "sentences": [sent.text.strip() for sent in doc.sents]
    }
