import fitz

pdf_path = "data/raw_papers/LoRA.pdf"

document = fitz.open(pdf_path)

print("Number of pages:", len(document))

for page_number, page in enumerate(document):
    text = page.get_text()

    print("\n" + "=" * 50)
    print("PAGE", page_number + 1)
    print("=" * 50)

    print(text[:1000])

document.close()