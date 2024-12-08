import os
from pdf_reader import read_pdf, clean_text
from nlp_processing import analyze_text

def analyze_pdfs(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        return print("Keine PDFs gefunden.")

    for pdf_path in pdf_files:
        print(f"\nAnalysiere: {os.path.basename(pdf_path)}")
        results = analyze_text(clean_text(read_pdf(pdf_path)))

        print("\nEntitäten:", *[f" - {entity} ({label})" for entity, label in results["entities"]], sep="\n")
        print("\nSätze:", *[f" - {sentence}" for sentence in results["sentences"][:5]], sep="\n")

if __name__ == "__main__":
    analyze_pdfs("./data")
