# Technische Dokumentation: 

## 1. Übersicht und Zweck

Dieses Dokument beschreibt das Python-Skript `app.py`, eine Streamlit-Anwendung zur automatisierten Bewertung der Antworten eines Informations-Chatbots. Das Hauptziel des Skripts ist es, die Qualität der vom Chatbot generierten Antworten zu messen, indem sie mit vordefinierten Referenzantworten verglichen werden.

Die Anwendung führt folgende Schritte aus:
1.  Lädt einen Testdatensatz mit Fragen und den dazugehörigen idealen Antworten (Referenztexten) aus einer CSV-Datei.
2.  Für jede Frage sucht es in einer Vektordatenbank, die aus den Referenztexten erstellt wurde, nach der relevantesten Antwort.
3.  Vergleicht die gefundene Antwort mit der erwarteten Referenzantwort mittels Kosinusähnlichkeit der Vektor-Embeddings.
4.  Bewertet die Antwort basierend auf dem Ähnlichkeits-Score in drei Kategorien: "Vollständig", "Teilweise" und "Unzureichend".
5.  Stellt die Ergebnisse in einer interaktiven Weboberfläche dar, inklusive einer Zusammenfassung, Filteroptionen und einer Exportfunktion für Excel.

## 2. Systemarchitektur

Das Skript ist als eine in sich geschlossene Webanwendung konzipiert und nutzt das Streamlit-Framework für die Benutzeroberfläche.

### 2.1. Komponenten

-   **Frontend (UI):** `Streamlit` wird verwendet, um eine interaktive Weboberfläche zu erstellen. Die UI zeigt den Titel, den Fortschritt, die Bewertungsergebnisse pro Frage, eine Gesamtübersicht und Download-Optionen an.
-   **Datenhaltung:** Eine CSV-Datei (`VuralZ_v1/testset.csv`) dient als primäre Datenquelle. Sie enthält die Testfälle.
-   **Sprachmodell (Embeddings):** `HuggingFaceEmbeddings` aus der `langchain`-Bibliothek lädt ein vortrainiertes Sentence-Transformer-Modell (`all-MiniLM-L6-v2`), um Text in hochdimensionale Vektoren (Embeddings) umzuwandeln.
-   **Vektordatenbank:** `FAISS` (Facebook AI Similarity Search) wird genutzt, um eine effiziente Vektordatenbank aus den Referenztexten zu erstellen. Dies ermöglicht eine schnelle Ähnlichkeitssuche.
-   **Logik-Schicht:** Der Kern des Skripts, geschrieben in Python, steuert den gesamten Prozess von Dateneinlesen über Verarbeitung bis zur Auswertung und Darstellung.

### 2.2. Datenfluss

1.  **Start:** Der Nutzer startet die Anwendung über `streamlit run app.py`.
2.  **Initialisierung:**
    -   Die Anwendung lädt die CSV-Datei `VuralZ_v1/testset.csv` in einen `pandas` DataFrame.
    -   Das Embedding-Modell wird initialisiert.
    -   Eine FAISS-Datenbank wird im Speicher (`in-memory`) aus den `referenztext`-Spalten des DataFrames erstellt.
3.  **Verarbeitungsschleife:**
    -   Die Anwendung iteriert über jede Zeile des DataFrames.
    -   Für jede `frage` wird eine Ähnlichkeitssuche (`similarity_search`) in der FAISS-DB durchgeführt, um die Top-k (hier k=10) ähnlichsten Dokumente zu finden.
    -   Eine `rerank`-Funktion verfeinert diese Ergebnisse, indem sie die Kosinusähnlichkeit zwischen der Frage und den zurückgegebenen Dokumenten berechnet und das Dokument mit der höchsten Ähnlichkeit auswählt.
    -   Die Embeddings der gefundenen Antwort (`top_doc`) und der erwarteten Antwort (`referenztext`) werden berechnet.
    -   Die Kosinusähnlichkeit zwischen diesen beiden Embeddings wird als Bewertungs-Score berechnet.
4.  **Ergebnisanzeige:**
    -   Der Score wird in eine der drei Bewertungskategorien eingeordnet.
    -   Alle Ergebnisse (Frage, Antwort, Erwartet, Score, Bewertung) werden auf der Streamlit-Oberfläche angezeigt.
    -   Eine zusammenfassende Statistik und eine detaillierte, filterbare Tabelle werden ebenfalls dargestellt.
5.  **Export:** Ein Download-Button ermöglicht den Export der gesamten Auswertung als Excel-Datei.

## 3. Technische Details

### 3.1. Abhängigkeiten

Das Skript erfordert folgende Python-Bibliotheken:
-   `streamlit`
-   `pandas`
-   `torch`
-   `langchain`
-   `faiss-cpu` (oder `faiss-gpu` für CUDA-Unterstützung)
-   `sentence-transformers`
-   `scikit-learn`
-   `xlsxwriter`
-   `openpyxl` (implizit für Pandas Excel-Funktionen)

Diese können über `pip install -r requirements.txt` installiert werden.

### 3.2. Wichtige Funktionen

-   `rerank(query, docs, embedding_model)`:
    -   **Zweck:** Verfeinert die Suchergebnisse der Vektordatenbank. Während die FAISS-DB Dokumente findet, die dem Frage-Embedding nahekommen, stellt diese Funktion sicher, dass das finale Ergebnis dasjenige ist, welches die höchste Kosinusähnlichkeit zur *ursprünglichen Frage* hat.
    -   **Input:** `query` (str), `docs` (Liste von `Document`-Objekten), `embedding_model`.
    -   **Output:** Eine Liste mit dem einen, am besten passenden `Document`.

-   `to_excel(df)`:
    -   **Zweck:** Konvertiert einen Pandas DataFrame in ein Excel-Dateiobjekt im Speicher.
    -   **Input:** `df` (Pandas DataFrame).
    -   **Output:** Ein `BytesIO`-Objekt, das die Excel-Datei enthält.

### 3.3. Konfiguration

-   **CSV-Pfad:** Der Pfad zur Testdatei ist in Zeile 16 hartcodiert: `csv_path = "VuralZ_v1/testset.csv"`. Für die Verwendung mit anderen Daten muss dieser Pfad angepasst werden.
-   **Embedding-Modell:** Das verwendete Modell ist in Zeile 23 festgelegt. Ein alternatives Modell ist auskommentiert. Durch Ändern des `model_name` kann die Genauigkeit und Performance beeinflusst werden.
-   **Bewertungsschwellen:** Die Grenzwerte für die Kategorien "Vollständig", "Teilweise" und "Unzureichend" sind in den Zeilen 93-98 definiert. Diese können je nach Anforderung und verwendetem Modell angepasst werden, um die Strenge der Bewertung zu justieren.

## 4. Verwendung und Ausführung

1.  **Voraussetzungen sicherstellen:**
    -   Stellen Sie sicher, dass Python und `pip` installiert sind.
    -   Installieren Sie alle Abhängigkeiten aus der `requirements.txt`.
2.  **Daten vorbereiten:**
    -   Erstellen Sie eine CSV-Datei mit den Spalten `frage` und `referenztext`.
    -   Stellen Sie sicher, dass die Datei unter dem im Skript angegebenen Pfad (`4_Semester/VuralZ_v1/testset.csv`) verfügbar ist oder passen Sie den Pfad im Skript an.
3.  **Anwendung starten:**
    -   Öffnen Sie ein Terminal im Wurzelverzeichnis des Projekts.
    -   Führen Sie den folgenden Befehl aus:
        ```bash
        streamlit run 4_Semester/app.py
        ```
    -   Die Anwendung wird in Ihrem Standard-Webbrowser geöffnet.

