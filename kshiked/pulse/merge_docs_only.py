"""
Merge only documents (images + PDFs) into a single PDF.
NO code files included.
"""

from pathlib import Path
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import PyPDF2


def merge_docs_to_pdf(source_dir: str, output_file: str):
    source_path = Path(source_dir)
    output_path = Path(output_file)
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    pdf_extensions = ['.pdf']
    exclude_folders = ['__pycache__', '.git']
    
    image_files = []
    pdf_files = []
    
    for file_path in sorted(source_path.rglob('*')):
        if file_path.is_file():
            if any(excl in file_path.parts for excl in exclude_folders):
                continue
            ext = file_path.suffix.lower()
            if ext in image_extensions:
                image_files.append(file_path)
            elif ext in pdf_extensions:
                pdf_files.append(file_path)
    
    print(f"Found {len(image_files)} images and {len(pdf_files)} PDFs")
    
    # Create PDF with images
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 15, 'KShield Pulse', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Documentation & Diagrams', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    
    # Add images
    for img_path in image_files:
        print(f"  Adding: {img_path.name}")
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, img_path.name, 0, 1, 'L')
        
        try:
            img = Image.open(img_path)
            # Scale to fit page
            max_w, max_h = 180, 250
            ratio = min(max_w / (img.width * 0.264583), max_h / (img.height * 0.264583))
            w_mm = img.width * 0.264583 * min(ratio, 1)
            
            temp = output_path.parent / f'_temp_{img_path.name}'
            img.save(temp)
            pdf.image(str(temp), w=w_mm)
            temp.unlink()
        except Exception as e:
            pdf.cell(0, 10, f'Error: {e}', 0, 1)
    
    # Save base PDF
    temp_pdf = output_path.parent / '_temp_base.pdf'
    pdf.output(str(temp_pdf))
    
    # Merge with existing PDFs
    if pdf_files:
        print(f"  Merging {len(pdf_files)} PDF(s)...")
        merger = PyPDF2.PdfMerger()
        merger.append(str(temp_pdf))
        for pf in pdf_files:
            print(f"    + {pf.name}")
            merger.append(str(pf))
        merger.write(str(output_path))
        merger.close()
        temp_pdf.unlink()
    else:
        temp_pdf.rename(output_path)
    
    print(f"\n✓ Created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    merge_docs_to_pdf(
        r"C:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\pulse",
        r"C:\Users\omegam\Downloads\kshield_pulse_docs.pdf"
    )
