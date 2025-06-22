# Technische Dokumentation: FH Technikum Wien Info Chatbot

> **Hinweis:** Dieses Dokument beschreibt den individuellen Prototyp von **Zeliha Vural**. Im Rahmen der Gruppenarbeit (Gruppe 04, 4. Semester, Inno2) hat jedes Gruppenmitglied einen eigenen Prototyp entwickelt.

## Inhaltsverzeichnis
1. [Projektübersicht](#1-projektübersicht)
2. [Systemarchitektur](#2-systemarchitektur)
3. [Technische Spezifikationen](#3-technische-spezifikationen)
4. [Installation und Setup](#4-installation-und-setup)
5. [Verwendung](#5-verwendung)
6. [API-Dokumentation](#6-api-dokumentation)
7. [Konfiguration](#7-konfiguration)
8. [Troubleshooting](#8-troubleshooting)
9. [Erweiterungen und Wartung](#9-erweiterungen-und-wartung)
10. [Glossar](#10-glossar)

---

## 1. Projektübersicht

### 1.1 Zweck und Zielsetzung
Das **FH Technikum Wien Info Chatbot**  ist eine automatisierte Bewertungsplattform zur Qualitätsmessung von Chatbot-Antworten. Das System vergleicht generierte Antworten mit vordefinierten Referenzantworten und bewertet deren Genauigkeit und Vollständigkeit.

### 1.2 Hauptfunktionen
- **Automatisierte Bewertung**: Vergleicht Chatbot-Antworten mit Referenzantworten
- **Vektor-basierte Suche**: Nutzt FAISS für effiziente Ähnlichkeitssuche
- **Interaktive Benutzeroberfläche**: Streamlit-basierte Web-UI
- **Detaillierte Analysen**: Score-Verteilung, Quellenanalyse, Performance-Metriken
- **Export-Funktionalität**: Excel-Export der Bewertungsergebnisse

### 1.3 Technologie-Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **NLP**: HuggingFace Transformers, Sentence-Transformers
- **Vektordatenbank**: FAISS
- **Datenverarbeitung**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **PDF-Verarbeitung**: PyMuPDF (fitz)

---

## 2. Systemarchitektur

### 2.1 Komponentendiagramm
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   PDF Loader    │    │  CSV Test Data  │
│   (Frontend)    │    │   (Data Input)  │    │   (Reference)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core Engine    │
                    │  (app.py)       │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  FAISS Vector   │    │  Embedding      │    │  Evaluation     │
│  Database       │    │  Model          │    │  Engine         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Datenfluss
1. **Initialisierung**: PDF-Dokumente laden → Text extrahieren → Chunks erstellen
2. **Vektorisierung**: Embeddings generieren → FAISS-Datenbank aufbauen
3. **Evaluation**: Testfragen verarbeiten → Ähnlichkeitssuche → Scoring
4. **Ausgabe**: Ergebnisse visualisieren → Export ermöglichen

### 2.3 Schlüsselkomponenten

#### 2.3.1 PDF-Verarbeitung (`load_pdf_documents`)
- **Zweck**: Extrahiert Text aus PDF-Dokumenten und erstellt verarbeitbare Chunks
- **Chunking-Strategie**: RecursiveCharacterTextSplitter mit 500 Zeichen Chunks
- **Metadaten**: Speichert Quelle, Seitennummer und Chunk-ID

#### 2.3.2 Vektordatenbank (FAISS)
- **Modell**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384-dimensional
- **Suchalgorithmus**: Cosine Similarity
- **Performance**: Optimiert für große Dokumentensammlungen

#### 2.3.3 Evaluierungsengine
- **Scoring-Methode**: Cosine Similarity zwischen Embeddings
- **Bewertungskategorien**: Vollständig (>0.75), Teilweise (0.45-0.75), Unzureichend (<0.45)
- **Kontext-Extraktion**: Intelligente Antwort-Extraktion aus gefundenem Text

---

## 3. Technische Spezifikationen

### 3.1 Systemanforderungen
- **Python**: 3.11 oder höher
- **RAM**: Mindestens 4GB (8GB empfohlen)
- **Speicherplatz**: 2GB für Modelle und Daten
- **Betriebssystem**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### 3.2 Abhängigkeiten
```python
# Core Dependencies
streamlit>=1.28.0
pandas>=2.0.0
torch>=2.0.0
langchain>=0.1.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0

# Data Processing
numpy>=1.24.0
PyMuPDF>=1.23.0

# Export & Utilities
xlsxwriter>=3.1.0
openpyxl>=3.1.0
```

### 3.3 Performance-Kennzahlen
- **PDF-Verarbeitung**: ~100 Seiten/Minute
- **Embedding-Generierung**: ~1000 Chunks/Minute
- **Ähnlichkeitssuche**: ~100 Queries/Sekunde
- **Speicherverbrauch**: ~2GB für 1000 PDF-Seiten

---

## 4. Installation und Setup

### 4.1 Voraussetzungen
```bash
# Python 3.11+ installieren
python --version

# Virtual Environment erstellen
python -m venv venv

# Virtual Environment aktivieren
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 4.2 Installation
```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Oder manuell installieren
pip install streamlit pandas torch langchain faiss-cpu sentence-transformers scikit-learn PyMuPDF xlsxwriter
```

### 4.3 Projektstruktur
```
4_Semester/
├── VuralZ_v1/
│   ├── app.py                          # Hauptanwendung
│   ├── Technische_Dokumentation.md     # Diese Dokumentation
│   ├── testset.csv                     # Testdaten
│   └── requirements.txt                # Abhängigkeiten
├── data/                              # PDF-Dokumente
│   ├── Richtlinie zur Einhebung, Rückerstattung, Befreiung von Beiträgen_V2.0.pdf
│   ├── Hausordnung 2024-05-15_V5.0.pdf
│   ├── 5 Satzungsteil Studienrechtliche Bestimmungen Prüfungsordnung 2024-06-13.pdf
│   └── Information über die Verwendung personenbezogener Daten von Studierenden.pdf
├── Documentation/                      # Projektdokumentation
├── Sprints/                           # Sprint-Protokolle
└── venv/                              # Virtual Environment
...
```

---

## 5. Verwendung

### 5.1 Anwendung starten
```bash
# Im Projektverzeichnis
cd 4_Semester/VuralZ_v1

# Streamlit-Anwendung starten
streamlit run app.py
```

### 5.2 Benutzeroberfläche
1. **Startseite**: Titel und Systemstatus
2. **Verarbeitung**: Automatisches Laden der PDF-Dokumente
3. **Bewertung**: Einzelne Fragen mit Scores und Bewertungen
4. **Zusammenfassung**: Statistiken und Metriken
5. **Export**: Download der Ergebnisse als Excel-Datei

### 5.3 Datenformat
#### CSV-Testdaten (`testset.csv`)
```csv
frage,referenztext
"Wie melde ich mich für eine Prüfung an?","Die Anmeldung erfolgt über das Online-Portal..."
"Was sind die Zulassungsvoraussetzungen?","Für die Zulassung benötigen Sie..."
```

#### PDF-Dokumente
- **Format**: PDF 1.4 oder höher
- **Sprache**: Deutsch (primär)
- **Struktur**: Text-basiert (keine reinen Bilder-PDFs)

---

## 6. API-Dokumentation

### 6.1 Hauptfunktionen

#### `load_pdf_documents(data_folder="../data")`
```python
def load_pdf_documents(data_folder="../data"):
    """
    Lädt und verarbeitet PDF-Dokumente aus dem angegebenen Ordner.
    
    Args:
        data_folder (str): Pfad zum Ordner mit PDF-Dateien
        
    Returns:
        list: Liste von Document-Objekten mit Text-Chunks
        
    Raises:
        Exception: Bei Fehlern beim Laden der PDF-Dateien
    """
```

#### `find_best_answer(query, referenz, embedding_model, db, k=50)`
```python
def find_best_answer(query, referenz, embedding_model, db, k=50):
    """
    Findet die beste Antwort durch Ähnlichkeitssuche.
    
    Args:
        query (str): Suchanfrage
        referenz (str): Referenzantwort
        embedding_model: HuggingFace Embedding-Modell
        db: FAISS-Datenbank
        k (int): Anzahl der zu durchsuchenden Dokumente
        
    Returns:
        tuple: (best_doc, similarity_score)
    """
```

#### `extract_exact_answer(found_text, reference_answer, max_context=200)`
```python
def extract_exact_answer(found_text, reference_answer, max_context=200):
    """
    Extrahiert die exakte Antwort aus dem gefundenen Text.
    
    Args:
        found_text (str): Gefundener Text
        reference_answer (str): Referenzantwort
        max_context (int): Maximale Kontextlänge
        
    Returns:
        str: Extrahierte Antwort
    """
```

### 6.2 Hilfsfunktionen

#### `clean_text_truncation(text, max_length=1500)`
```python
def clean_text_truncation(text, max_length=1500):
    """
    Schneidet Text sauber ab, ohne mitten im Satz zu enden.
    
    Args:
        text (str): Zu kürzender Text
        max_length (int): Maximale Länge
        
    Returns:
        str: Gekürzter Text
    """
```

---

## 7. Konfiguration

### 7.1 Modell-Konfiguration
```python
# Embedding-Modell ändern
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Alternative: "all-mpnet-base-v2"
)
```

### 7.2 Bewertungsschwellen anpassen
```python
# In der Hauptlogik (Zeile ~200)
if similarity_score > 0.75:        # Vollständig
    bewertung = "🟢 Vollständig"
elif similarity_score >= 0.45:     # Teilweise
    bewertung = "🟡 Teilweise"
else:                              # Unzureichend
    bewertung = "🔴 Unzureichend"
```

### 7.3 Chunking-Parameter
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Chunk-Größe in Zeichen
    chunk_overlap=100,     # Überlappung zwischen Chunks
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### 7.4 Suchparameter
```python
# Anzahl der zu durchsuchenden Dokumente
k=50  # Höhere Werte = mehr Präzision, aber langsamer
```

---

## 8. Troubleshooting

### 8.1 Häufige Probleme

#### Problem: "Keine PDF-Dokumente gefunden"
**Lösung:**
- Überprüfen Sie den Pfad in `data_folder="../data"`
- Stellen Sie sicher, dass PDF-Dateien im Ordner vorhanden sind
- Prüfen Sie Dateiberechtigungen

#### Problem: "Out of Memory" Fehler
**Lösung:**
- Reduzieren Sie `chunk_size` auf 300-400
- Verringern Sie `k` in der Ähnlichkeitssuche
- Schließen Sie andere Anwendungen

#### Problem: Langsame Performance
**Lösung:**
- Verwenden Sie GPU-Version von FAISS (`faiss-gpu`)
- Reduzieren Sie die Anzahl der PDF-Dokumente
- Optimieren Sie Chunking-Parameter

### 8.2 Debugging
```python
# Debug-Ausgaben aktivieren
import logging
logging.basicConfig(level=logging.DEBUG)

# Speicherverbrauch überwachen
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
```

### 8.3 Logs und Monitoring
- **Streamlit-Logs**: `streamlit run app.py --logger.level debug`
- **System-Monitoring**: Task Manager / Activity Monitor
- **Performance-Metriken**: In der UI unter "Score-Verteilung"

---

## 9. Erweiterungen und Wartung


### 9.1 Wartungsaufgaben
- **Regelmäßige Updates**: HuggingFace-Modelle aktualisieren
- **Performance-Optimierung**: Chunking-Parameter anpassen
- **Datenqualität**: PDF-Dokumente auf Konsistenz prüfen
- **Backup**: Testdaten und Konfigurationen sichern

### 9.2 Skalierung
```python
# Für große Dokumentensammlungen
# 1. Chunking optimieren
chunk_size = 1000  # Größere Chunks
chunk_overlap = 200

# 2. FAISS-Index optimieren
import faiss
index = faiss.IndexFlatIP(384)  # Inner Product für bessere Performance

# 3. Batch-Verarbeitung
def process_batch(queries, batch_size=10):
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        # Verarbeite Batch
```

---

## 10. Glossar

### 10.1 Technische Begriffe
- **Embedding**: Vektor-Darstellung von Text in einem hochdimensionalen Raum
- **FAISS**: Facebook AI Similarity Search - Bibliothek für effiziente Ähnlichkeitssuche
- **Chunking**: Aufteilung langer Texte in kleinere, verarbeitbare Einheiten
- **Cosine Similarity**: Maß für die Ähnlichkeit zwischen zwei Vektoren
- **Reranking**: Nachträgliche Sortierung von Suchergebnissen

### 10.2 Bewertungskategorien
- **Vollständig (🟢)**: Score > 0.75 - Antwort entspricht vollständig der Referenz
- **Teilweise (🟡)**: Score 0.45-0.75 - Antwort ist teilweise korrekt
- **Unzureichend (🔴)**: Score < 0.45 - Antwort ist ungenügend

### 10.3 Metriken
- **Score**: Ähnlichkeitswert zwischen 0 und 1
- **Durchschnittlicher Score**: Arithmetisches Mittel aller Scores
- **Quellenanalyse**: Verteilung der verwendeten Dokumente
- **Performance**: Verarbeitungszeit und Speicherverbrauch

---

## 11. Kontakt und Support

### 11.1 Entwicklerin
- **Projekt**: FH Technikum Wien Info Chatbot
- **Semester**: 4. Semester
- **Gruppe**: Gruppe 04 (Inno2)
- **Entwicklerin**: Zeliha Vural
- **Hinweis**: Individueller Prototyp im Rahmen der Gruppenarbeit - jedes Gruppenmitglied hat einen eigenen Prototyp entwickelt

### 11.2 Dokumentation
- **Version**: 1.0
- **Letzte Aktualisierung**: Juni 2025


### 11.3 Lizenz
Dieses Projekt ist Teil des FH Technikum Wien Curriculums und unterliegt den entsprechenden akademischen Richtlinien.

---

*Diese Dokumentation wird kontinuierlich aktualisiert und erweitert. Für Fragen oder Verbesserungsvorschläge wenden Sie sich an Zeliha Vural.* 