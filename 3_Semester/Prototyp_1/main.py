import os
from pdf_reader import read_pdf, clean_text
from nlp_processing import analyze_text

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
            print(f"Extrahierter Text (erste 500 Zeichen): {raw_text[:1500]}...\n")
            cleaned_text = clean_text(raw_text)
            print(f"Bereinigter Text (erste 500 Zeichen): {cleaned_text[:10000]}...\n")
            results = analyze_text(cleaned_text)
            print("\nEntitäten:")
            print(*[f" - {entity} ({label})" for entity, label in results["entities"]], sep="\n")

            print("\nSätze:")
            print(*[f" - {sentence}" for sentence in results["sentences"][:5]], sep="\n")  # Nur die ersten 5 Sätze

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {os.path.basename(pdf_path)}: {e}")

if __name__ == "__main__":
    folder_path = "./data"
    analyze_pdfs(folder_path)