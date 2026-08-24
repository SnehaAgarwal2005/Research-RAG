import pymupdf
import os

input_folder = "data/raw_papers"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(input_folder, filename)

        print("Processing:", filename)

        document = pymupdf.open(pdf_path)

        all_text = ""

        for page_number, page in enumerate(document):

            text = page.get_text()

            all_text += f"\n\n--- PAGE {page_number + 1} ---\n\n"
            all_text += text

        document.close()

        output_filename = filename.replace(".pdf", ".txt")

        output_path = os.path.join(
            output_folder,
            output_filename
        )

        with open(output_path, "w", encoding="utf-8") as file:

            file.write(all_text)

        print("Saved:", output_filename)

print("\nAll papers processed successfully!")