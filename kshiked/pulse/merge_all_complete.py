"""
Merge ALL documents from pulse folder (including subdirectories).
Includes: text files, images, and PDFs (embedded as links or base64).
Excludes: HTML files
"""

import os
import base64
from pathlib import Path
from datetime import datetime


def get_mime_type(extension):
    """Get MIME type for files."""
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.pdf': 'application/pdf',
    }
    return mime_types.get(extension.lower(), 'application/octet-stream')


def merge_all_documents(source_dir: str, output_file: str):
    """
    Merge all documents into a single HTML file.
    Recursively scans subdirectories.
    """
    source_path = Path(source_dir)
    output_path = Path(output_file)
    
    # Categorize files
    text_extensions = ['.txt', '.py', '.mermaid', '.md', '.json', '.csv', '.env']
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
    pdf_extensions = ['.pdf']
    exclude_extensions = ['.html']
    exclude_folders = ['__pycache__', '.git', 'node_modules']
    
    text_files = []
    image_files = []
    pdf_files = []
    
    # Recursively find all files
    for file_path in sorted(source_path.rglob('*')):
        if file_path.is_file():
            # Skip files in excluded folders
            if any(excl in file_path.parts for excl in exclude_folders):
                continue
            
            ext = file_path.suffix.lower()
            if ext in exclude_extensions:
                continue
            elif ext in image_extensions:
                image_files.append(file_path)
            elif ext in pdf_extensions:
                pdf_files.append(file_path)
            elif ext in text_extensions or ext == '':
                text_files.append(file_path)
    
    print(f"Found:")
    print(f"  - {len(text_files)} text files")
    print(f"  - {len(image_files)} image files")
    print(f"  - {len(pdf_files)} PDF files")
    
    # Build HTML
    html_parts = []
    
    # HTML Header
    html_parts.append(f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KShield Pulse - Complete Documentation</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; border-bottom: 3px solid #58a6ff; padding-bottom: 10px; }}
        h2 {{ color: #58a6ff; margin-top: 40px; }}
        h3 {{ color: #8b949e; }}
        .file-section {{ background: #161b22; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #30363d; }}
        .file-header {{ background: linear-gradient(135deg, #238636, #1f6feb); color: white; padding: 12px 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; font-weight: bold; }}
        .file-path {{ font-size: 0.8em; opacity: 0.7; }}
        pre {{ background: #0d1117; color: #c9d1d9; padding: 15px; overflow-x: auto; border-radius: 4px; font-size: 12px; border: 1px solid #30363d; max-height: 600px; overflow-y: auto; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #30363d; margin: 10px 0; border-radius: 4px; }}
        .toc {{ background: #161b22; padding: 20px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #30363d; }}
        .toc ul {{ columns: 2; list-style: none; padding-left: 10px; }}
        .toc li {{ margin: 8px 0; }}
        .meta {{ color: #8b949e; font-size: 0.9em; }}
        .pdf-embed {{ width: 100%; height: 800px; border: 1px solid #30363d; border-radius: 4px; }}
        a {{ color: #58a6ff; }}
        .category {{ background: #21262d; padding: 10px 15px; border-radius: 6px; margin: 30px 0 15px 0; border-left: 4px solid #58a6ff; }}
    </style>
</head>
<body>
    <h1>🛡️ KShield Pulse - Complete Documentation</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    Source: {source_path}<br>
    Total files: {len(text_files) + len(image_files) + len(pdf_files)}</p>
    
    <div class="toc">
        <h2>📑 Table of Contents</h2>
        <h3>PDFs ({len(pdf_files)})</h3>
        <ul>
''')
    
    for fp in pdf_files:
        rel_path = fp.relative_to(source_path)
        html_parts.append(f'            <li>📕 {rel_path}</li>\n')
    
    html_parts.append('        </ul>\n        <h3>Images (' + str(len(image_files)) + ')</h3>\n        <ul>\n')
    for fp in image_files:
        rel_path = fp.relative_to(source_path)
        html_parts.append(f'            <li>🖼️ {rel_path}</li>\n')
    
    html_parts.append('        </ul>\n        <h3>Text/Code Files (' + str(len(text_files)) + ')</h3>\n        <ul>\n')
    for fp in text_files:
        rel_path = fp.relative_to(source_path)
        html_parts.append(f'            <li>📝 {rel_path}</li>\n')
    
    html_parts.append('        </ul>\n    </div>\n')
    
    # Add PDFs section
    if pdf_files:
        html_parts.append('    <div class="category"><h2>📕 PDF Documents</h2></div>\n')
        for pdf_path in pdf_files:
            print(f"  Embedding PDF: {pdf_path.name}")
            rel_path = pdf_path.relative_to(source_path)
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_data = base64.b64encode(f.read()).decode('utf-8')
                html_parts.append(f'''
    <div class="file-section">
        <div class="file-header">📕 {pdf_path.name}<br><span class="file-path">{rel_path}</span></div>
        <embed src="data:application/pdf;base64,{pdf_data}" type="application/pdf" class="pdf-embed">
        <p><a href="data:application/pdf;base64,{pdf_data}" download="{pdf_path.name}">⬇️ Download PDF</a></p>
    </div>
''')
            except Exception as e:
                html_parts.append(f'    <p>Error loading {pdf_path.name}: {e}</p>\n')
    
    # Add images section
    if image_files:
        html_parts.append('    <div class="category"><h2>🖼️ Images & Diagrams</h2></div>\n')
        for img_path in image_files:
            print(f"  Embedding image: {img_path.name}")
            rel_path = img_path.relative_to(source_path)
            try:
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                mime = get_mime_type(img_path.suffix)
                html_parts.append(f'''
    <div class="file-section">
        <div class="file-header">🖼️ {img_path.name}<br><span class="file-path">{rel_path}</span></div>
        <img src="data:{mime};base64,{img_data}" alt="{img_path.name}">
    </div>
''')
            except Exception as e:
                html_parts.append(f'    <p>Error loading {img_path.name}: {e}</p>\n')
    
    # Add text files section
    if text_files:
        html_parts.append('    <div class="category"><h2>📝 Source Code & Text Files</h2></div>\n')
        for txt_path in text_files:
            print(f"  Adding text: {txt_path.name}")
            rel_path = txt_path.relative_to(source_path)
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                # Escape HTML
                content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_parts.append(f'''
    <div class="file-section">
        <div class="file-header">📝 {txt_path.name}<br><span class="file-path">{rel_path}</span></div>
        <pre>{content}</pre>
    </div>
''')
            except Exception as e:
                html_parts.append(f'    <p>Error loading {txt_path.name}: {e}</p>\n')
    
    # HTML Footer
    html_parts.append('''
    <hr style="border-color: #30363d; margin-top: 50px;">
    <p class="meta" style="text-align: center;">End of Document</p>
</body>
</html>
''')
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    file_size = output_path.stat().st_size
    print(f"\n✓ Created: {output_path}")
    print(f"  Size: {file_size / (1024*1024):.2f} MB")
    
    return output_path


if __name__ == "__main__":
    # Source is the entire pulse folder
    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse"
    OUTPUT_FILE = r"C:\Users\omegam\Downloads\kshield_pulse_complete.html"
    
    print("=" * 60)
    print("MERGING ALL PULSE FILES (PDF + IMAGES + CODE)")
    print("=" * 60)
    
    merge_all_documents(SOURCE_DIR, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("COMPLETE! Open in browser:")
    print(OUTPUT_FILE)
    print("=" * 60)
