import spacy
import json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from itertools import chain
from transformers import pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

try:
    nlp = spacy.load("de_core_news_lg")
except Exception as e:
    raise ImportError(f"Modell konnte nicht geladen werden: {e}")


def deduplicate_entities(entities):
    entity_summary = defaultdict(lambda: {"count": 0, "positions": []})
    for ent in entities:
        key = (ent.text, ent.label_)
        entity_summary[key]["count"] += 1
        entity_summary[key]["positions"].append((ent.start_char, ent.end_char))
    return [
        (ent_text, ent_label, data["count"], data["positions"])
        for (ent_text, ent_label), data in entity_summary.items()
    ]

def extract_paragraphs(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    raw_paragraphs = text.split('\n\n')
    paragraphs = [
        para.strip().replace('\n', ' ')
        for para in raw_paragraphs
        if len(para.strip()) > 50
    ]
    return paragraphs

def group_sentences(sentences, group_size=1):
    return [
        " ".join(sentences[i:i+group_size])
        for i in range(0, len(sentences), group_size)
        if len(" ".join(sentences[i:i+group_size])) > 50  # avoid short junk chunks
    ]

def extract_sentences(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    joined_text = ' '.join(lines)
    pattern = re.compile(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])')
    raw_sentences = pattern.split(joined_text)
    cleaned_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) > 30 and not sentence.lower() in ['-', '•', '']:
            cleaned_sentences.append(sentence)
    return cleaned_sentences

# Neue robuste Chunk-Funktion

def extract_chunks(text):
    """
    Kombiniert spaCy-Satztrennung mit Heuristik zur Aufspaltung von Bulletpoints oder Aufzählungen.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    joined_text = ' '.join(lines)
    doc = nlp(joined_text)
    spacy_sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    for sent in spacy_sentences:
        # Split bullets or entries starting with dashes, bullets, etc.
        if re.search(r"[•\-]", sent) or re.match(r"^[A-ZÄÖÜa-zäöüß]+\s*:", sent):
            parts = re.split(r"(?<!\w)-\s+|[•]", sent)
            for part in parts:
                part = part.strip()
                if len(part) > 30:
                    chunks.append(part)
        else:
            chunks.append(sent)

    return chunks

def embed_sentences(sentences, source_name):
    formatted_sentences = [f"passage: {s.strip()}" for s in sentences]
    embeddings = sentence_model.encode(formatted_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    return [(sentence, embedding, source_name) for sentence, embedding in zip(sentences, embeddings)]

def find_best_match(user_question, embedded_sentences, top_k=35):
    """
    Finds the best matching sentence chunks for a given user query using cosine similarity.
    Optimized for E5-style bi-encoder embedding setup.
    """
    # Encode query
    query_embedding = sentence_model.encode(f"query: {user_question.strip()}", convert_to_tensor=True)

    # Compute similarities
    scored = []
    for sentence, sent_emb, source in embedded_sentences:
        sim = util.cos_sim(query_embedding, sent_emb)[0][0].item()
        scored.append(((sentence, sent_emb, source), sim))

    # Sort by similarity
    top_matches = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    # remove near duplicates (cosine sim > 0.95 between candidates)
    unique = []
    seen_embeddings = []
    for candidate, score in top_matches:
        sentence, emb, source = candidate
        if all(util.cos_sim(emb, seen)[0][0] < 0.95 for seen in seen_embeddings):
            seen_embeddings.append(emb)
            unique.append((candidate, score))

    return unique

def expand_query(query: str, expansion_file="query_expansion.json") -> str:
    expanded_query = query.lower()
    try:
        with open(expansion_file, "r", encoding="utf-8") as f:
            expansions = json.load(f)
    except Exception as e:
        print(f" Fehler beim Laden von Query-Expansion-Datei: {e}")
        return query
    for keyword, synonyms in expansions.items():
        if keyword in expanded_query:
            expanded_query += " " + " ".join(synonyms)
    return expanded_query

# Pipeline für Paraphrases
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", device=0 if torch.cuda.is_available() else -1)

def generate_paraphrases(text, num_return_sequences=5):
    input_text = f"paraphrase: {text} </s>"
    outputs = paraphraser(
        input_text,
        max_length=100,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=1.2,
    )
    return list({x["generated_text"].strip() for x in outputs})


def filter_best_paraphrases(original, paraphrases, top_n=8):
    orig_emb = sentence_model.encode(f"query: {original}", convert_to_tensor=True)
    para_embs = sentence_model.encode([f"query: {p}" for p in paraphrases], convert_to_tensor=True)

    scores = util.cos_sim(orig_emb, para_embs)[0].tolist()
    scored_paras = sorted(zip(paraphrases, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored_paras[:top_n]]