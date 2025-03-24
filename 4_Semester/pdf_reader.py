import fitz  # PyMuPDF
import re
from spacy.lang.de.stop_words import STOP_WORDS

def read_pdf(file_path):
    """
    Einfache Textextraktion mit PyMuPDF.
    """
    try:
        with fitz.open(file_path) as doc:
            extracted_text = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")  # Text extrahieren
               # print(f"Seite {page_num}: {text[:500]}")  # Debug-Ausgabe für jede Seite
                extracted_text.append(text)
            return "\n".join(extracted_text)
    except Exception as e:
        raise ValueError(f"Fehler beim Lesen der PDF '{file_path}': {e}")


def clean_text(text):
    """
    Bereinigt den Text durch Entfernen von Seitenzahlen und unnötigen Sonderzeichen,
    während relevante Zahlen und Inhalte beibehalten werden.
    """
    # Entferne typische Muster für Seitenzahlen und Seitencounter
    text = re.sub(r'\b(seite\s+\d+(/\d+)?)\b', '', text, flags=re.IGNORECASE)  # "Seite X" oder "Seite X/Y"
    text = re.sub(r'\b(\d+\s+von\s+\d+)\b', '', text, flags=re.IGNORECASE)  # "X von Y"
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Entfernt isolierte Zahlenzeilen

    # Reduziere mehrfach vorkommende Leerzeichen
    text = re.sub(r'\s+', ' ', text)

    # Entferne Sonderzeichen außer Punkt und Komma
    text = re.sub(r'[^\w\s.,]', '', text)

    # Entferne Bindestriche, die Wörter trennen
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Konvertiere Text in Kleinbuchstaben
    text = text.lower()

    # Ersetze Synonyme
    synonyms = {
        "fhtw": "fachhochschule technikum wien",
    }
    for synonym, replacement in synonyms.items():
        text = re.sub(rf'\b{synonym}\b', replacement, text)

    # Stopwörter entfernen (nur auf Wortebene, nicht auf Zahlen)
    tokens = text.split()
    filtered_tokens = [
        token for token in tokens if token.lower() not in STOP_WORDS or token.isdigit()
    ]

    return " ".join(filtered_tokens).strip()
