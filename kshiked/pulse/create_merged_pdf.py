"""
Merge all documents from pulse folder into a single PDF file.
Includes: text files, images, and PDFs.
Excludes: HTML files
"""

import os
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import PyPDF2
import io


class MergedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'KShield Pulse - Complete Documentation', 0, 0, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def add_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def add_section_header(self, text):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(51, 51, 51)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, text, 0, 1, 'L', fill=True)
        self.ln(3)
        
    def add_code(self, code_text):
        self.set_font('Courier', '', 8)
        self.set_text_color(0, 0, 0)
        # Handle encoding
        safe_text = code_text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 4, safe_text)
        self.ln(5)


def merge_to_pdf(source_dir: str, output_file: str):
    """
    Merge all documents into a single PDF file.
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
    print()
    
    # Create PDF
    pdf = MergedPDF()
    pdf.add_page()
    
    # Title page
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 102, 204)
    pdf.ln(40)
    pdf.cell(0, 20, 'KShield Pulse', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Complete Documentation', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.cell(0, 6, f'Total files: {len(text_files) + len(image_files) + len(pdf_files)}', 0, 1, 'C')
    
    # Table of contents
    pdf.add_page()
    pdf.add_title('Table of Contents')
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, f'Images ({len(image_files)})', 0, 1)
    pdf.set_font('Helvetica', '', 9)
    for fp in image_files:
        rel_path = fp.relative_to(source_path)
        pdf.cell(0, 5, f'  - {rel_path}', 0, 1)
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, f'Text/Code Files ({len(text_files)})', 0, 1)
    pdf.set_font('Helvetica', '', 9)
    for fp in text_files:
        rel_path = fp.relative_to(source_path)
        pdf.cell(0, 5, f'  - {rel_path}', 0, 1)
    
    # Add images
    if image_files:
        pdf.add_page()
        pdf.add_title('Images & Diagrams')
        
        for img_path in image_files:
            print(f"  Adding image: {img_path.name}")
            try:
                rel_path = img_path.relative_to(source_path)
                pdf.add_section_header(str(rel_path))
                
                # Resize image if needed
                img = Image.open(img_path)
                max_width = 180  # mm
                img_width_mm = img.width * 0.264583  # pixels to mm
                
                if img_width_mm > max_width:
                    scale = max_width / img_width_mm
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save temp image
                temp_path = Path(output_path.parent) / f'temp_{img_path.name}'
                img.save(temp_path)
                
                pdf.image(str(temp_path), w=min(180, img.width * 0.264583))
                pdf.ln(10)
                
                # Clean up
                temp_path.unlink()
                
            except Exception as e:
                pdf.set_font('Helvetica', 'I', 9)
                pdf.cell(0, 5, f'Error loading image: {e}', 0, 1)
    
    # Add text files
    if text_files:
        pdf.add_page()
        pdf.add_title('Source Code & Text Files')
        
        for txt_path in text_files:
            print(f"  Adding text: {txt_path.name}")
            try:
                rel_path = txt_path.relative_to(source_path)
                pdf.add_section_header(str(rel_path))
                
                with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Truncate very long files
                if len(content) > 15000:
                    content = content[:15000] + '\n\n... [TRUNCATED - File too large] ...'
                
                pdf.add_code(content)
                
            except Exception as e:
                pdf.set_font('Helvetica', 'I', 9)
                pdf.cell(0, 5, f'Error: {e}', 0, 1)
    
    # Save generated PDF
    temp_output = output_path.parent / 'temp_merged.pdf'
    pdf.output(str(temp_output))
    print(f"\n  Generated base PDF")
    
    # If there are existing PDFs to merge, combine them
    if pdf_files:
        print(f"  Merging {len(pdf_files)} existing PDF(s)...")
        merger = PyPDF2.PdfMerger()
        
        # Add our generated PDF first
        merger.append(str(temp_output))
        
        # Add existing PDFs
        for pdf_file in pdf_files:
            print(f"    Adding: {pdf_file.name}")
            try:
                merger.append(str(pdf_file))
            except Exception as e:
                print(f"    Error adding {pdf_file.name}: {e}")
        
        # Write final merged PDF
        merger.write(str(output_path))
        merger.close()
        
        # Clean up temp file
        temp_output.unlink()
    else:
        # Just rename temp to final
        temp_output.rename(output_path)
    
    file_size = output_path.stat().st_size
    print(f"\n{'='*50}")
    print(f"PDF Created: {output_path}")
    print(f"Size: {file_size / (1024*1024):.2f} MB")
    print(f"{'='*50}")
    
    return output_path


if __name__ == "__main__":
    SOURCE_DIR = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse"
    OUTPUT_FILE = r"C:\Users\omegam\Downloads\kshield_pulse_complete.pdf"
    
    print("=" * 50)
    print("CREATING MERGED PDF")
    print("=" * 50)
    print()
    
    merge_to_pdf(SOURCE_DIR, OUTPUT_FILE)
