import os
import pickle
from pdf_reader import read_pdf, clean_text
from nlp_processing import analyze_text, extract_paragraphs, extract_sentences, embed_sentences, find_best_match
from nlp_processing import expand_query


def analyze_pdfs(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print("Keine PDFs im Ordner gefunden.")
        return

    cache_path = "pickle_cache/sentence_cache.pkl"
    all_sentences = load_embeddings_from_cache(cache_path)

    if not all_sentences:
        print("\n Keine gecachten Embeddings gefunden – verarbeite PDFs...")
        all_sentences = []

        for pdf_path in pdf_files:
            print(f"\nAnalysiere: {os.path.basename(pdf_path)}")

            try:
                raw_text = read_pdf(pdf_path)
                #print(f"\nExtrahierter Text (erste 500 Zeichen): {raw_text[:500]}...\n")

                cleaned_text = clean_text(raw_text)
                #print(f"\nBereinigter Text (erste 500 Zeichen): {cleaned_text[:500]}...\n")

                results = analyze_text(cleaned_text)
                #print(f"\nEntitäten für {os.path.basename(pdf_path)}:")
                #for entity, label, count, positions in results["entities"]:
                    #print(f" - {entity} ({label}) - {count} mal, Positionen: {positions}")

                print("\nVollständige Sätze/Paragraphen:")

                # use extract_sentences or extract_paragraphs

                chunksSen = extract_sentences(raw_text)
                #chunksPar = extract_paragraphs(raw_text)

                all_sentences.extend(embed_sentences(chunksSen, os.path.basename(pdf_path)))
                print(*[f" - {sentence}" for sentence in chunksSen], sep="\n")  # Alle Sätze ausgeben
                #all_sentences.extend(embed_sentences(chunksPar, os.path.basename(pdf_path)))
                #print(*[f" - {sentence}" for sentence in chunksPar], sep="\n")  # Alle Paragraphen ausgeben

            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {os.path.basename(pdf_path)}: {e}")
        
        # Cache embeddings
        save_embeddings_to_cache(all_sentences, cache_path)

    # Loop for user queries
    while True:
        query = input("\nFrage an den Bot (oder 'exit'): ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        expanded = expand_query(query)
        matches = find_best_match(expanded, all_sentences)

        matches = [m for m in matches if m[1] > 0.15]

        if matches:
            best_match = matches[0]
            print("\n💡 Beste Antwort:")
            print(f"{best_match[0][0]}  [📄 {best_match[0][2]} | 🔍 Score: {best_match[1]:.2f}]")

            print("\n\n Weitere relevante Antworten:\n")
            for (sentence, _, source), score in matches:
                print(f"- {sentence}  [📄 {source} | 🔍 Score: {score:.2f}]")
        else:
            print("⚠️ Keine passenden Antworten gefunden.")

def save_embeddings_to_cache(data, cache_path="pickle_cache/sentence_cache.pkl"):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f" Embeddings gespeichert in {cache_path}")


def load_embeddings_from_cache(cache_path="pickle_cache/sentence_cache.pkl"):
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f" Geladene gecachte Embeddings aus {cache_path}")
        return data
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "data")
    analyze_pdfs(folder_path)
