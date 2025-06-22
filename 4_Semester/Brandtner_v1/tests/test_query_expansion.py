
from nlp_processing import expand_query

def test_expand_simple_term():
    query = "plagiat"
    expanded = expand_query(query)
    assert "plagiatsprüfung" in expanded
    assert "plagiatstool" in expanded

def test_expand_unknown_term():
    query = "nichtvorhanden"
    expanded = expand_query(query)
    assert expanded == "nichtvorhanden"
