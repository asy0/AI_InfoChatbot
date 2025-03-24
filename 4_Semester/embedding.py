import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pdf_reader import read_pdf, clean_text
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lade SBERT Modell für semantische Suche
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Lade T5 Modell für Antwortgenerierung
t5_model_name = "t5-small"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Pfad zum PDF-Dokument
script_dir = os.path.dirname(__file__)
folder_path = os.path.join(script_dir, "data")
document_path = os.path.join(folder_path, "Information über die Verwendung personenbezogener Daten von Studierenden.pdf")

# Überprüfe, ob das Dokument existiert
if not os.path.exists(document_path):
    raise FileNotFoundError(f" Fehler: Die Datei {document_path} wurde nicht gefunden!")

def load_and_embed_document(file_path):
    """Liest das PDF, bereinigt den Text und erstellt Embeddings für Absätze."""
    try:
        raw_text = read_pdf(file_path)
        cleaned_text = clean_text(raw_text)
        paragraphs = cleaned_text.split(".\n")  # Absätze trennen

        if not paragraphs:
            raise ValueError(f"Error: No paragraphs found in document {file_path}!")

        paragraph_embeddings = [sbert_model.encode(para, convert_to_numpy=True) for para in paragraphs]
        return paragraphs, np.array(paragraph_embeddings)

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise

# Lade und verarbeite das Dokument
paragraphs, paragraph_embeddings = load_and_embed_document(document_path)

# FAISS Index für schnellere Suche erstellen
d = paragraph_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(paragraph_embeddings)

def create_search_index(embeddings):
    """Creates a FAISS index for efficient similarity search."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def search_relevant_text(query, paragraphs, embeddings, top_k=3):
    """Searches for the most relevant paragraphs based on the user query."""
    try:
        # Create search index
        index = create_search_index(embeddings)
        
        # Encode query
        query_embedding = sbert_model.encode(query, convert_to_numpy=True)
        
        # Search for similar paragraphs
        D, I = index.search(np.array([query_embedding]), top_k)
        
        results = []
        for idx, distance in zip(I[0], D[0]):
            if idx != -1:
                results.append({
                    'text': paragraphs[idx],
                    'similarity_score': float(1 / (1 + distance))  # Convert distance to similarity score
                })
        
        return results if results else []

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return []

def generate_answer(query, context):
    """Generates an answer based on the most relevant paragraphs."""
    try:
        if not context:
            return "Ich konnte dazu leider keine genaue Information in den Dokumenten finden."

        # Combine context paragraphs
        combined_context = " ".join([item['text'] for item in context])
        
        # Prepare input for T5
        input_text = f"Antwort auf: {query} Kontext: {combined_context}"
        input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate answer
        output_ids = t5_model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            temperature=0.7,
            top_p=0.9
        )
        
        return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Entschuldigung, es ist ein Fehler bei der Antwortgenerierung aufgetreten."

# Beispiel-Fragen
test_questions = [
    "Wer ist für die Verarbeitung der personenbezogenen Daten von Studierenden verantwortlich?",
    "Welche Kontaktdaten hat der Datenschutzbeauftragte der FHTW?",
    "Unter welchen Voraussetzungen verarbeitet die FHTW die personenbezogenen Daten von Studierenden?",
    "Welche Zwecke hat die Verarbeitung personenbezogener Daten an der FHTW?",
    "Wie lange werden die personenbezogenen Daten von Studierenden gespeichert?",
    "Wer erhält personenbezogene Daten von Studierenden neben der FHTW?",
    "Welche Rechte haben Studierende gemäß der DSGVO in Bezug auf ihre personenbezogenen Daten?",
    "Unter welchen Umständen kann eine Person eine Löschung ihrer personenbezogenen Daten beantragen?",
    "Für welchen Zweck wird Videoüberwachung an der FHTW eingesetzt?",
    "Was passiert mit den Daten nach dem Abschluss des Studiums?"
]

for question in test_questions:
    answer = generate_answer(question, search_relevant_text(question, paragraphs, paragraph_embeddings))
    print(f"\n🔹 **Frage:** {question}\n **Antwort:** {answer}\n")
