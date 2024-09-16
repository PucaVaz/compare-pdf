# PDF Comparison Tool Documentation

## Overview
The main goal of this python  Python script compares two PDF documents of resolutions from the Conselho Superior de Ensino, Pesquisa e Extensão (Consepe) at the Universidade Federal da Paraíba (UFPB), but they differ in their content and purpose.
so, my job this to identifies changes between them, and generates an HTML diff view. It uses natural language processing techniques to compare text segments and visualize the differences.

## Dependencies

- PyMuPDF (fitz)
- Transformers
- PyTorch
- scikit-learn
- NumPy
- difflib
- re
- html

## Main Components

### 1. Device Configuration

```python
def define_device():
    # ... (function implementation)
```

This function determines the available hardware (CUDA GPU, Apple Silicon, or CPU) for processing.

### 2. Model Loading

```python
def load_model(model_name='sentence-transformers/all-MiniLM-L6-v2', device=None):
    # ... (function implementation)
```

Loads a pre-trained language model and tokenizer for text embedding.

### 3. PDF Text Extraction

```python
def extract_text_from_pdf(pdf_path):
    # ... (function implementation)
```

Extracts text content from a PDF file.

### 4. Text Segmentation

```python
def segment_text(text):
    # ... (function implementation)
```

Segments the extracted text into chunks based on article numbers.

### 5. Embedding Computation

```python
def compute_embeddings(text_chunks, tokenizer, model, device):
    # ... (function implementation)
```

Computes embeddings for each text chunk using the loaded model.

### 6. Similarity Calculation

```python
def compute_similarity_matrix(embeddings1, embeddings2):
    # ... (function implementation)
```

Computes the cosine similarity between embeddings of original and updated text chunks.

### 7. Match Finding

```python
def find_matches(similarity_matrix, threshold=0.9):
    # ... (function implementation)
```

Identifies matching chunks based on similarity scores.

### 8. HTML Diff Generation

```python
def generate_diff_html(original_chunks, updated_chunks, matches, output_path='diff_view.html'):
    # ... (function implementation)
```

Generates an HTML file visualizing the differences between the original and updated PDFs.

## Main Execution Flow

The `main()` function orchestrates the entire process:

1. Configure the device
2. Load the model and tokenizer
3. Extract text from both PDFs
4. Segment the extracted text
5. Compute embeddings for both texts
6. Calculate similarity matrix
7. Find matches based on similarity
8. Generate and save the HTML diff view

## Usage
Download all the python dependencies
```bash
pip install -r requirements.txt
```
and then, run it. 
```bash
python main.py
```
## Output

The script generates an HTML file named "diff_view.html" which provides a visual representation of the differences between the two PDFs. The output includes:

- Removed chunks (highlighted in red)
- Updated chunks (with line-by-line differences)
- Unchanged chunks
- Similarity scores for each chunk
- Summary of changes (lines added/removed)

This HTML file can be opened in any web browser for easy viewing and analysis of the differences between the two PDF documents.
