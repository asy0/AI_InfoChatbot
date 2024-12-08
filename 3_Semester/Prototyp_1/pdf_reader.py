import fitz
from fitz import open as fitz_open


def read_pdf(file_path):
    return " ".join(page.get_text() for page in fitz.open(file_path))

def clean_text(text):
    return " ".join(text.split())
