
from nlp_processing import find_best_match, embed_sentences

def test_find_best_match_ordering():
    embedded = embed_sentences(["Plagiatsprüfung erfolgt mit Software", "Essen ist lecker"], "dummy.pdf")
    query = "Wie wird Plagiat geprüft?"
    results = find_best_match(query, embedded)
    assert results[0][0][0].lower().startswith("plagiatsprüfung")
    assert results[0][1] > results[1][1]
