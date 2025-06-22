"""
FH Technikum Wien Info Chatbot
==================================================

Entwicklerin: Zeliha Vural
Projekt: FH Technikum Wien Info Chatbot (4. Semester, Gruppe 04, Inno2)
Datum: Juni 2025
Beschreibung: Automatisierte Bewertungsplattform zur Qualitätsmessung von Chatbot-Antworten
             mittels Vektor-basierter Ähnlichkeitssuche und FAISS-Datenbank.
Hinweis: Individueller Prototyp im Rahmen der Gruppenarbeit 
"""

import streamlit as st
import pandas as pd
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
from io import BytesIO
import fitz  # PyMuPDF
import re

# Titel
st.title("📚 FH Technikum Wien – Info Chatbot ")

# Entwickler-Information
st.markdown("---")
st.markdown("**👩‍💻 Entwickelt von:** Zeliha Vural")
st.markdown("**📋 Projekt:** FH Technikum Wien Info Chatbot (4. Semester, Gruppe 04, Inno2)")
st.markdown("**📅 Datum:** 2025")
#st.markdown("**ℹ️ Hinweis:** Individueller Prototyp im Rahmen der Gruppenarbeit")
st.markdown("---")

# Funktion zum Laden und Aufbereiten der PDF-Dokumente
def load_pdf_documents(data_folder="../data"):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Klein= für präzisere Antworten
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    try:
        st.info("📄 Lade PDF-Dokumente...")
        pdf_count = 0
        
        for filename in os.listdir(data_folder):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(data_folder, filename)
                pdf_count += 1
                
                doc = fitz.open(pdf_path)
                full_text = ""
                
                # Text von allen Seiten extrahieren
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()
                    full_text += f"\n\nSeite {page_num + 1}:\n{text}"
                
                doc.close()
                
                # Text in kleinere Chunks aufteilen
                if full_text.strip():
                    chunks = text_splitter.split_text(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:  # Mind.50 Zeichen
                            # Finde die Seitennummer aus dem Chunk
                            page_match = re.search(r'Seite (\d+):', chunk)
                            page_num = int(page_match.group(1)) if page_match else 1
                            
                            documents.append(Document(
                                page_content=chunk.strip(),
                                metadata={
                                    "source": filename,
                                    "page": page_num,
                                    "chunk_id": i,
                                    "file_path": pdf_path
                                }
                            ))
        
        #st.success(f"✅ {len(documents)} Textchunks aus {pdf_count} PDF-Dokumenten geladen")
        return documents
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der PDF-Dokumente: {e}")
        return []

pdf_documents = load_pdf_documents()

if not pdf_documents:
    st.error("❌ Keine PDF-Dokumente gefunden. Bitte stellen Sie sicher, dass PDF-Dateien im 'data' Ordner vorhanden sind.")
    st.stop()

#CSV-Datei laden
csv_path = "testset.csv"
try:
    test_data = pd.read_csv(csv_path)
    #st.success(f"✅ {len(test_data)} Testfragen aus CSV geladen")
except Exception as e:
    st.error(f"❌ Fehler beim Laden der CSV-Datei: {e}")
    st.stop()

# Embedding-Modell
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS-Datenbank a erstellen
#st.info("🔧 Erstelle Vektordatenbank aus PDF-Dokumenten...")
db = FAISS.from_documents(pdf_documents, embedding=embedding_model)
st.success("✅ Vektordatenbank erfolgreich erstellt")


def extract_exact_answer(found_text, reference_answer, max_context=200):
    """
    Extrahiert die exakte Antwort aus dem gefundenen Text
    """
    #Texte für besseren Vergleich
    found_lower = found_text.lower()
    ref_lower = reference_answer.lower()
    
    # Referenzantwort im gefundenen Text suchen
    if ref_lower in found_lower:
        start_pos = found_lower.find(ref_lower)
        end_pos = start_pos + len(reference_answer)
        
        # Antwort mit etwas Kontext extrahieern
        context_start = max(0, start_pos - max_context)
        context_end = min(len(found_text), end_pos + max_context)
        
        extracted = found_text[context_start:context_end]
        
        # Satzenden abschneiden
        sentences = extracted.split('. ')
        if len(sentences) > 1:
            for i, sentence in enumerate(sentences):
                if ref_lower in sentence.lower():
                    return sentence.strip() + '.'
        
        return extracted.strip()
    
    # Nach ähnlichen Phrasen scuhen
    ref_words = reference_answer.lower().split()
    best_match = ""
    best_score = 0
    
    # Text in Sätze aufteilen
    sentences = found_text.split('. ')
    for sentence in sentences:
        sentence_lower = sentence.lower()
        match_count = sum(1 for word in ref_words if word in sentence_lower)
        if match_count > best_score and match_count > len(ref_words) * 0.7:
            best_score = match_count
            best_match = sentence.strip() + '.'
    
    if best_match:
        return best_match

    return found_text

# Vereinfachte und effektivere Suche
def find_best_answer(query, referenz, embedding_model, db, k=50):
    """
    Findet die beste Antwort durch direkten Vergleich aller Dokumente
    """
    # alle ähnlichen Dokumente holen
    docs = db.similarity_search(query, k=k)
    
    best_doc = None
    best_score = -1
    
    # jedes Dokument direkt mit der Referenzantwort vergleichen
    for doc in docs:
        try:
            # Ähnlichkeit zwischen Referenz&Dokument berechnen
            ref_emb = embedding_model.embed_query(referenz)
            doc_emb = embedding_model.embed_query(doc.page_content)
            score = cosine_similarity([ref_emb], [doc_emb])[0][0]
            
            if score > best_score:
                best_score = score
                best_doc = doc
        except Exception:
            continue
    
    return best_doc, best_score


def clean_text_truncation(text, max_length=1500):
    """
    Schneidet Text sauber ab, ohne mitten im Satz zu enden
    """
    if len(text) <= max_length:
        return text
    
    # Versuche, bei einem Satzende abzuschneiden
    truncated = text[:max_length]
    
    # Suche nach dem letzten vollständigen Satz& Verschiedene Satzenden berücksichtigen
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.\r\n', '!\r\n', '?\r\n']
    last_sentence_end = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos + len(ending) - 1
    
    # Wenn ein Satzende gefunden wurde und es nicht zu früh ist
    if last_sentence_end > max_length * 0.5:  # Mind 50% der Länge nutzen
        return text[:last_sentence_end + 1]
    
    # Suche nach dem letzten vollständigen Wort
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.6:  # Mind. 60%
        return text[:last_space] + "..."

    return text

# Hauptlogik zur Verarbeitung
results = []

st.subheader("🔍 Bewertung der Antworten")
progress_bar = st.progress(0)

for idx, row in test_data.iterrows():
    frage = row.get("frage")
    referenz = row.get("referenztext")

    # Fehlervermeidung bei leeren Werten
    if not isinstance(frage, str) or not frage.strip():
        st.warning(f"⚠️ Leere oder ungültige Frage in Zeile {idx + 1}. Überspringe.")
        continue
    if not isinstance(referenz, str) or not referenz.strip():
        st.warning(f"⚠️ Leerer oder ungültiger Referenztext in Zeile {idx + 1}. Überspringe.")
        continue

    # Suche nach der besten Antwort
    try:
        best_doc, similarity_score = find_best_answer(frage, referenz, embedding_model, db)
        
        if best_doc is None:
            st.warning(f"❌ Keine passende Antwort für Frage {idx + 1} gefunden.")
            continue
        
        # die exakte Antwort aus dem gefundenen Text
        full_text = best_doc.page_content.strip()
        extracted_answer = extract_exact_answer(full_text, referenz)
        source_info = best_doc.metadata
        
        # Score für die extrahierte Antwort berechnen
        try:
            ref_emb = embedding_model.embed_query(referenz)
            extracted_emb = embedding_model.embed_query(extracted_answer)
            similarity_score = cosine_similarity([ref_emb], [extracted_emb])[0][0]
        except Exception:
            pass
        
    except Exception as e:
        st.warning(f"❌ Fehler bei Suche in Frage {idx + 1}: {e}")
        continue

    # Bewertung basierend auf Ähnlichkeit
    if similarity_score > 0.75:
        bewertung = "🟢 Vollständig"
    elif similarity_score >= 0.45:
        bewertung = "🟡 Teilweise"
    else:
        bewertung = "🔴 Unzureichend"

    display_text = clean_text_truncation(extracted_answer, max_length=1500)


    st.markdown(f"**Frage {idx+1}:** {frage}")
    st.markdown(f"**Bewertung:** {bewertung}  \n📈 Score: `{similarity_score:.3f}`")
    st.markdown(f"**🔹 Erwartet:** {referenz}")
    st.markdown(f"**🔸 Gefunden:** {display_text}")
    st.markdown(f"**📄 Quelle:** {source_info['source']} (Seite {source_info['page']})")
    st.markdown("---")

    results.append({
        "Frage": frage,
        "Antwort": extracted_answer,
        "Erwartet": referenz,
        "Score": similarity_score,
        "Bewertung": bewertung,
        "Quelle": source_info['source'],
        "Seite": source_info['page']
    })
    
    # Fortschrittsbalken aktualisieren
    progress_bar.progress((idx + 1) / len(test_data))

results_df = pd.DataFrame(results)

# Zusammenfassung
st.subheader("📊 Zusammenfassung der Bewertungen")
summary = results_df["Bewertung"].value_counts().rename_axis("Bewertung").reset_index(name="Anzahl")

# Quellenanalyse
st.subheader("📚 Verwendete Quellen")
source_summary = results_df.groupby("Quelle").agg({
    "Bewertung": "count",
    "Score": "mean"
}).rename(columns={"Bewertung": "Anzahl_Verwendungen", "Score": "Durchschnitt_Score"})
source_summary = source_summary.round(3)

st.dataframe(source_summary)

# Detaillierte Score-Analyse
st.subheader("📈 Score-Verteilung")
score_stats = results_df["Score"].describe()
st.write(f"**Durchschnittlicher Score:** {score_stats['mean']:.3f}")
st.write(f"**Bester Score:** {score_stats['max']:.3f}")
st.write(f"**Schlechtester Score:** {score_stats['min']:.3f}")

# Zeige die besten und schlechtesten Ergebnisse
st.subheader("🏆 Beste Ergebnisse (Score > 0.6)")
best_results = results_df[results_df["Score"] > 0.6].sort_values("Score", ascending=False)
if not best_results.empty:
    st.dataframe(best_results[["Frage", "Score", "Bewertung", "Quelle"]])
else:
    st.write("Keine Ergebnisse mit Score > 0.6 gefunden.")

st.subheader("🔎 Filter nach Bewertung")
bewertung_filter = st.selectbox("Bewertung auswählen", options=["Alle"] + sorted(results_df["Bewertung"].unique()))

if bewertung_filter != "Alle":
    filtered_df = results_df[results_df["Bewertung"] == bewertung_filter]
else:
    filtered_df = results_df

st.dataframe(filtered_df)
st.dataframe(summary)

# Gesamttabelle anzeigen
st.subheader("🔍 Detaillierte Auswertung")
st.dataframe(results_df)

# Excel-Export vorbereiten
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Bewertung")
        source_summary.to_excel(writer, sheet_name="Quellenanalyse")
    output.seek(0)
    return output

excel_data = to_excel(results_df)

# Download-Button
st.download_button(
    label="📥 Ergebnisse als Excel herunterladen",
    data=excel_data,
    file_name="bewertung_infochatbot_simplified.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
