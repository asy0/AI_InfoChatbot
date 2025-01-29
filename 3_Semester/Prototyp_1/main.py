import os
from pdf_reader import read_pdf, clean_text
from nlp_processing import analyze_text, extract_complete_sentences


def analyze_pdfs(folder_path):
    # Alle PDF-Dateien im Ordner finden
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print("Keine PDFs im Ordner gefunden.")
        return

    for pdf_path in pdf_files:
        print(f"\nAnalysiere: {os.path.basename(pdf_path)}")

        try:
            raw_text = read_pdf(pdf_path)
            print(f"\nExtrahierter Text (erste 500 Zeichen): {raw_text[:500]}...\n")
            cleaned_text = clean_text(raw_text)
            print("\n-------------------")
            print(f"\nBereinigter Text (erste 500 Zeichen): {cleaned_text[:500]}...\n")
            results = analyze_text(cleaned_text)
            print("\n-------------------")
            print(f"\nEntitäten für {os.path.basename(pdf_path)}:")
            for entity, label, count, positions in results["entities"]:
                print(f" - {entity} ({label}) - {count} mal, Positionen: {positions}")

            print("\n-------------------")
            print("\nVollständige Sätze:")
            complete_sentences = extract_complete_sentences(raw_text)
            print(*[f" - {sentence}" for sentence in complete_sentences[:5]], sep="\n")  # Nur die ersten 5 Sätze

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {os.path.basename(pdf_path)}: {e}")


if __name__ == "__main__":
    # Ordnerpfad relativ zum aktuellen Skript
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "data")
    analyze_pdfs(folder_path)
