import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib
import re
import html

def define_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA for GPU acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS for Apple Silicon Mac")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_model(model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model
 
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def segment_text(text):
    return re.findall(r'Art\. \d+º?.*?(?=(?:Art\. \d+º?)|\Z)', text, re.DOTALL)

def compute_embeddings(text_chunks, tokenizer, model, device):
    embeddings = []
    with torch.no_grad():
        for chunk in text_chunks:
            inputs = tokenizer(
                chunk,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model(**inputs, return_dict=True)
            
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(chunk_embedding)
    
    return np.vstack(embeddings)

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


def generate_diff_html(original_chunks, updated_chunks, matches, output_path='diff_view.html'):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Comparison Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .chunk {
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
            }
            .chunk-header {
                font-weight: bold;
                padding: 10px;
                background-color: #f0f0f0;
                border-bottom: 1px solid #ddd;
            }
            .chunk-content {
                padding: 10px;
                white-space: pre-wrap;
            }
            .removed {
                background-color: #ffecec;
            }
            .added {
                background-color: #eaffea;
            }
            .unchanged {
                background-color: #ffffff;
            }
            .diff-line {
                margin: 0;
                padding: 2px 0;
            }
            .diff-removed {
                background-color: #ffd7d5;
            }
            .diff-added {
                background-color: #d7ffd5;
            }
            .similarity {
                float: right;
                font-weight: normal;
            }
            .diff-summary {
                font-style: italic;
                margin-bottom: 10px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>PDF Comparison Results</h1>
    """

    for i, (orig_idx, matched_idx, similarity) in enumerate(matches):
        if matched_idx is None:
            html_content += f"""
            <div class="chunk removed">
                <div class="chunk-header">Removed Chunk {i}</div>
                <div class="chunk-content">{html.escape(original_chunks[orig_idx])}</div>
            </div>
            """
        else:
            diff = list(difflib.unified_diff(original_chunks[orig_idx].splitlines(), 
                                             updated_chunks[matched_idx].splitlines(), 
                                             lineterm=''))
            
            diff_html = ""
            lines_removed = 0
            lines_added = 0
            for line in diff[3:]:  # Skip the first three lines (including the hunk header)
                if line.startswith('+'):
                    diff_html += f'<p class="diff-line diff-added">{html.escape(line)}</p>'
                    lines_added += 1
                elif line.startswith('-'):
                    diff_html += f'<p class="diff-line diff-removed">{html.escape(line)}</p>'
                    lines_removed += 1
                else:
                    diff_html += f'<p class="diff-line">{html.escape(line)}</p>'

            status = "unchanged" if similarity >= 0.99 else "added"
            html_content += f"""
            <div class="chunk {status}">
                <div class="chunk-header">
                    {"Unchanged" if status == "unchanged" else "Updated"} Chunk {i}
                    <span class="similarity">Similarity: {similarity:.2f}</span>
                </div>
                <div class="diff-summary">
                    Changes: {lines_removed} line(s) removed, {lines_added} line(s) added
                </div>
                <div class="chunk-content">{diff_html}</div>
            </div>
            """

    html_content += """
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Enhanced diff view generated and saved to {output_path}.")


def main():
    device = define_device()
    tokenizer, model = load_model(device=device)

    pdf_path_original = "B.pdf"
    pdf_path_updated = "A.pdf"
    
    print("Extracting text from PDFs...")
    text_original = extract_text_from_pdf(pdf_path_original)
    text_updated = extract_text_from_pdf(pdf_path_updated)

    print("Segmenting text into chunks...")
    chunks_original = segment_text(text_original)
    chunks_updated = segment_text(text_updated)

    print("Computing embeddings on original text...")
    embeddings_original = compute_embeddings(chunks_original, tokenizer, model, device)
    print("Computing embeddings on updated text...")
    embeddings_updated = compute_embeddings(chunks_updated, tokenizer, model, device)

    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings_original, embeddings_updated)

    print("Finding matches based on similarity scores...")
    matches = find_matches(similarity_matrix)

    print("Generating HTML diff view...")
    generate_diff_html(chunks_original, chunks_updated, matches)

if __name__ == "__main__":
    main()