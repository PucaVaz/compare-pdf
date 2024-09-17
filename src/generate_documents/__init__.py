# Imports to the pdf 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import red, green, black, grey
# Imports to the html
import difflib
import html
from reportlab.lib.pagesizes import letter

def generate_diff_pdf(original_chunks, updated_chunks, matches, output_path='pdf_diff_result.pdf'):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['Heading1']
    normal_style = styles['BodyText']
    removed_style = ParagraphStyle('Removed', parent=normal_style, textColor=red)
    added_style = ParagraphStyle('Added', parent=normal_style, textColor=green)
    unchanged_style = ParagraphStyle('Unchanged', parent=normal_style, textColor=grey)

    story.append(Paragraph("PDF Comparison Results", title_style))
    story.append(Spacer(1, 12))

    for i, (orig_idx, matched_idx, similarity) in enumerate(matches):
        if matched_idx is None:
            story.append(Paragraph(f"Removed Chunk {i}", styles['Heading2']))
            story.append(Paragraph(original_chunks[orig_idx], removed_style))
        else:
            status = "Unchanged" if similarity >= 0.99 else "Updated"
            story.append(Paragraph(f"{status} Chunk {i} (Similarity: {similarity:.2f})", styles['Heading2']))
            
            diff = list(difflib.unified_diff(original_chunks[orig_idx].splitlines(), 
                                             updated_chunks[matched_idx].splitlines(), 
                                             lineterm=''))
            
            lines_removed = 0
            lines_added = 0
            for line in diff[3:]:  # Skip the first three lines (including the hunk header)
                if line.startswith('+'):
                    story.append(Paragraph(line, added_style))
                    lines_added += 1
                elif line.startswith('-'):
                    story.append(Paragraph(line, removed_style))
                    lines_removed += 1
                else:
                    story.append(Paragraph(line, unchanged_style))
            
            story.append(Paragraph(f"Changes: {lines_removed} line(s) removed, {lines_added} line(s) added", normal_style))
        
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF comparison results generated and saved to {output_path}.")


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
