
from nlp_processing import extract_chunks
from pdf_reader import clean_text

def test_extract_chunks_bullets(bullet_sample_text):
    chunks = extract_chunks(bullet_sample_text)

    assert len(chunks) == 2

    first, second = chunks

    assert "bachelor" in first.lower()
    assert "einhaltung" in first.lower()
    assert "behörden" in second.lower()
    assert "anlassfall" in second.lower()

def test_clean_text_removes_page_markers():
    raw = "Seite 3 von 5\nBachelor- und Masterarbeiten"
    cleaned = clean_text(raw)
    assert "seite" not in cleaned
    assert "bachelor" in cleaned
