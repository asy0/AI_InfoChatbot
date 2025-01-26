import spacy
from collections import defaultdict

try:
    nlp = spacy.load("de_core_news_lg")
except Exception as e:
    raise ImportError("Modell konnte nicht geladen werden: {e}")

def analyze_text(text):
    """
   Analysiert den bereinigten Text und extrahiert Entitäten und tokenisierte Sätze.
    """
    doc = nlp(text)

 # Debugging: Zeige alle erkannten Entitäten mit Positionen im Text
   # for ent in doc.ents:
   #     print(f"Debug: {ent.text}, Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")

        
    # Tokenisierte Sätze extrahieren
    tokenized_sentences = [sent.text.strip() for sent in doc.sents]


    # Entferne Duplikate bei Entitäten, fasse sie zusammen
    unique_entities = deduplicate_entities(doc.ents)
   

    return {
        "entities": unique_entities,
        "tokenized_sentences": tokenized_sentences,
        #"complete_sentences": complete_sentences
    }

def deduplicate_entities(entities):
    """
    Entfernt Duplikate basierend auf dem Text, Label und Position der Entitäten und gibt zusammengefasste Entitäten zurück.
    """
    entity_summary = defaultdict(lambda: {"count": 0, "positions": []})

    for ent in entities:
        # Nutzt (Text, Label) als Schlüssel und speichert zusätzlich die Start- und Endposition
        key = (ent.text, ent.label_)
        entity_summary[key]["count"] += 1
        entity_summary[key]["positions"].append((ent.start_char, ent.end_char))
       
    # Rückgabe der zusammengefassten Entitäten mit Anzahl und Positionen
    return [
        (ent_text, ent_label, data["count"], data["positions"])
        for (ent_text, ent_label), data in entity_summary.items()
    ]

def extract_complete_sentences(text):
    """
    Extrahiert vollständige Sätze aus dem Originaltext.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]
