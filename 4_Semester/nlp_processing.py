import spacy
import json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

try:
    nlp = spacy.load("de_core_news_lg")
except Exception as e:
    raise ImportError("Modell konnte nicht geladen werden: {e}")

def analyze_text(text):
    doc = nlp(text)
    tokenized_sentences = [sent.text.strip() for sent in doc.sents]
    unique_entities = deduplicate_entities(doc.ents)
    return {
        "entities": unique_entities,
        "tokenized_sentences": tokenized_sentences
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

def extract_paragraphs(text):
    """
    Splits the input text into paragraphs for more context-aware semantic search.
    Filters out short or meaningless chunks.
    """
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split on double line breaks (common paragraph delimiter in PDFs)
    raw_paragraphs = text.split('\n\n')

    # Clean and filter
    paragraphs = [
        para.strip().replace('\n', ' ')
        for para in raw_paragraphs
        if len(para.strip()) > 50  # skip very short chunks
    ]

    return paragraphs


def extract_sentences(text):
    """
    Splits by period, question marks, or linebreaks after full sentences.
    Keeps bullets and avoids breaking between bullet header + content.
    """

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive whitespace and join lines that are part of the same sentence
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    joined_text = ' '.join(lines)

    # Regex pattern to split at end of sentences while keeping bullet points like "•", "", "-"
    pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')  # split after . ! ? followed by uppercase

    # Split text and clean
    raw_sentences = pattern.split(joined_text)
    cleaned_sentences = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) > 30 and not sentence.lower() in ['-', '•', '']:  # filter short/junk lines
            cleaned_sentences.append(sentence)

    return cleaned_sentences


# Load multilingual model (works well for German)
sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Store embeddings per sentence with metadata
def embed_sentences(sentences, source_name):
    """
    Takes a list of sentences and returns a list of (sentence, embedding, source) tuples.
    """
    embeddings = sentence_model.encode(sentences)
    return [(sentence, embedding, source_name) for sentence, embedding in zip(sentences, embeddings)]

def find_best_match(user_question, embedded_sentences, top_k=5):
    question_embedding = sentence_model.encode([user_question])[0]
    scored = []

    for sentence, sent_emb, source in embedded_sentences:
        score = cosine_similarity([question_embedding], [sent_emb])[0][0]

        # Penalize generic short sentences
        if len(sentence) < 50:
            score *= 0.9
        elif "rechte" in sentence.lower() and len(sentence) < 100:
            score *= 0.85

        # Boost: keyword match from raw user query
        for keyword in user_question.lower().split():
            if keyword in sentence.lower():
                score += 0.02  # small boost per keyword match

        scored.append(((sentence, sent_emb, source), score))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def expand_query(query: str, expansion_file="query_expansion.json") -> str:
    """
    Expands the query using synonyms from an external JSON file.
    """
    expanded_query = query.lower()

    try:
        with open(expansion_file, "r", encoding="utf-8") as f:
            expansions = json.load(f)
    except Exception as e:
        print(f" Fehler beim Laden von Query-Expansion-Datei: {e}")
        return query  # fallback: original query

    for keyword, synonyms in expansions.items():
        if keyword in expanded_query:
            expanded_query += " " + " ".join(synonyms)

    return expanded_query