import fitz  # PyMuPDF
import re
#from fitz import open as fitz_open

def read_pdf(file_path):
    """
    Liest den gesamten Text aus einer PDF-Datei.
    """
    try:
        with fitz.open(file_path) as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        raise ValueError(f"Fehler beim Lesen der PDF '{file_path}': {e}")

def clean_text(text):
    """
    Bereinigt den Text durch Entfernen von Sonderzeichen und Reduktion von Leerzeichen.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,]', '', text)  # Entfernen Sie Sonderzeichen außer Punkt und Komma
    return text.strip()








