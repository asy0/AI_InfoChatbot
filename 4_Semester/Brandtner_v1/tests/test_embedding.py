
from nlp_processing import embed_sentences
import numpy as np

def test_embed_sentences_format():
    sentences = ["Das ist ein Test.", "Noch ein Satz."]
    results = embed_sentences(sentences, "testsource.pdf")
    assert len(results) == 2
    assert isinstance(results[0][0], str)
    assert isinstance(results[0][1], np.ndarray)
    assert results[0][2] == "testsource.pdf"
