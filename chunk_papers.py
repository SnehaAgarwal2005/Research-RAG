import json
import os
import re

# Folder containing extracted text files
input_folder = "data/processed"

# Folder where chunks will be saved
output_folder = "data/chunks"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Settings
CHUNK_SIZE = 350
OVERLAP = 50

all_chunks = []
chunk_id = 0


def split_into_chunks(text):
    """
    Split text into smaller overlapping pieces.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        start = end - OVERLAP

    return chunks


# Process every TXT file
for filename in os.listdir(input_folder):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(input_folder, filename)

    print("Processing:", filename)

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    paper_name = filename.replace(".txt", ".pdf")

    # Find page sections
    pages = re.split(r"--- PAGE (\d+) ---", text)

    # The split creates:
    # [text_before, page_number, page_text, page_number, page_text...]

    current_page = 1

    # Process page sections
    for i in range(1, len(pages), 2):

        page_number = int(pages[i])

        page_text = pages[i + 1]

        # Split this page into chunks
        page_chunks = split_into_chunks(page_text)

        for number, chunk_text in enumerate(page_chunks, start=1):

            if not chunk_text.strip():
                continue

            chunk = {
                "chunk_id": chunk_id,
                "paper": paper_name,
                "page": page_number,
                "chunk_number": number,
                "text": chunk_text
            }

            all_chunks.append(chunk)

            chunk_id += 1


# Save chunks
output_file = os.path.join(output_folder, "chunks.json")

with open(output_file, "w", encoding="utf-8") as file:

    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 60)
print("Total chunks created:", len(all_chunks))
print("Saved:", output_file)
print("=" * 60)