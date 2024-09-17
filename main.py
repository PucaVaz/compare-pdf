import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Local imports
from src.generate_documents import generate_diff_html, generate_diff_pdf
from src.embedding_system.embedding_chain import ComputeEmbeddings
 
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def segment_text(text):
    return re.findall(r'Art\. \d+º?.*?(?=(?:Art\. \d+º?)|\Z)', text, re.DOTALL)

def compute_similarity_matrix(embeddings1, embeddings2):
    return cosine_similarity(embeddings1, embeddings2)

def find_matches(similarity_matrix, threshold=0.9):
    matches = []
    for i, row in enumerate(similarity_matrix):
        max_similarity = max(row)
        if max_similarity > threshold:
            matched_index = np.argmax(row)
            matches.append((i, matched_index, max_similarity))
        else:
            matches.append((i, None, max_similarity))
    return matches


def main():
    # Define the computer embeddings
    embedding_system = ComputeEmbeddings()

    pdf_path_original = "B.pdf"
    pdf_path_updated = "A.pdf"
    
    print("Extracting text from PDFs...")
    text_original = extract_text_from_pdf(pdf_path_original)
    text_updated = extract_text_from_pdf(pdf_path_updated)

    print("Segmenting text into chunks...")
    chunks_original = segment_text(text_original)
    chunks_updated = segment_text(text_updated)

    print("Computing embeddings on original text...")
    embeddings_original = embedding_system.compute_embeddings(chunks=chunks_original)
    print("Computing embeddings on updated text...")
    embeddings_updated = embedding_system.compute_embeddings(chunks=chunks_updated)

    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings_original, embeddings_updated)

    print("Finding matches based on similarity scores...")
    matches = find_matches(similarity_matrix)

    print("Generating HTML diff view...")
    generate_diff_html(chunks_original, chunks_updated, matches)

    print("Generating PDF diff result...")
    generate_diff_pdf(chunks_original, chunks_updated, matches)

if __name__ == "__main__":
    main()