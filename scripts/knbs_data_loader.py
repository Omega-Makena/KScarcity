import pandas as pd
from pathlib import Path
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

class KnbsDataLoader:
    def __init__(self, data_dir="data/knbs_reports"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            print(f"Warning: {self.data_dir} does not exist.")
            
    def list_reports(self):
        """Returns a DataFrame of available reports with metadata parsed from filenames."""
        files = []
        for f in self.data_dir.glob("*.[pP][dD][fF]"):
            meta = self._parse_filename(f.name)
            meta['path'] = str(f)
            meta['filename'] = f.name
            files.append(meta)
            
        for f in self.data_dir.glob("*.[xX][lL][sS]*"):
             meta = self._parse_filename(f.name)
             meta['path'] = str(f)
             meta['filename'] = f.name
             meta['type'] = 'excel'
             files.append(meta)

        return pd.DataFrame(files)

    def _parse_filename(self, filename):
        """Extracts report type, month, year from filename."""
        # Normalize
        name = filename.lower().replace("-", " ").replace("_", " ")
        
        # 1. Extract Year (4 digits 20xx)
        year_match = re.search(r'(20\d{2})', name)
        year = int(year_match.group(1)) if year_match else None
        
        # 2. Extract Month
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }
        month = None
        for m_name, m_num in months.items():
            if m_name in name:
                month = m_num
                break
                
        # 3. Classify Type
        report_typ = "unknown"
        if "cpi" in name or "consumer price" in name or "inflation" in name:
            report_typ = "cpi_inflation"
        elif "leading economic" in name or "lei" in name:
            report_typ = "leading_indicators"
        elif "gdp" in name or "gross domestic" in name:
            report_typ = "gdp"
        elif "balance of payment" in name:
            report_typ = "balance_of_payments"
            
        return {
            "year": year,
            "month": month,
            "report_type": report_typ
        }

    def extract_text_from_pdf(self, filepath, pages=None):
        """Extracts text from a PDF file using pypdf or pdfplumber."""
        text = ""
        filepath = Path(filepath)
        limit = pages if pages else 3
        
        # Try pypdf first (preferred/available)
        if HAS_PYPDF:
            try:
                reader = pypdf.PdfReader(filepath)
                for i in range(min(limit, len(reader.pages))):
                    text += reader.pages[i].extract_text() or ""
                    text += "\n"
                return text
            except Exception as e:
                logger.warning(f"pypdf failed on {filepath}: {e}")
        
        # Fallback to pdfplumber
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(filepath) as pdf:
                    for i, page in enumerate(pdf.pages):
                        if i >= limit: break
                        text += page.extract_text() or ""
                        text += "\n"
                return text
            except Exception as e:
                logger.error(f"pdfplumber failed on {filepath}: {e}")
        
        if not HAS_PYPDF and not HAS_PDFPLUMBER:
             logger.error("No PDF library installed (pypdf or pdfplumber).")
             
        return text

    def get_latest_inflation(self):
        """
        Scans downloaded reports for the latest CPI/Inflation report 
        and extracts the overall year-on-year inflation rate.
        """
        df = self.list_reports()
        if df.empty:
            return None
            
        # Filter for inflation reports
        cpi_df = df[df['report_type'] == 'cpi_inflation'].sort_values(by=['year', 'month'], ascending=False)
        
        for _, row in cpi_df.iterrows():
            filepath = row['path']
            try:
                # Search first 5 pages (summary is usually on page 3)
                text = self.extract_text_from_pdf(filepath, pages=5)
                if not text:
                    continue
                
                # Regex patterns for inflation
                # Note: "per cent" is sometimes split or inconsistent, so we match loosely
                patterns = [
                    r"year-on-year inflation.*?stood at\s+(\d+\.?\d*)",
                    r"annual consumer price inflation.*?was\s+(\d+\.?\d*)",
                    r"overall year-on-year inflation.*?was\s+(\d+\.?\d*)",
                    r"inflation rate.*?was\s+(\d+\.?\d*)\s+per cent"
                ]
                
                for pat in patterns:
                    match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
                    if match:
                        val = float(match.group(1))
                        logger.info(f"Found inflation {val}% in {row['filename']}")
                        return val
                        
            except Exception as e:
                logger.warning(f"Failed to parse {filepath}: {e}")
                continue
                
        return None

if __name__ == "__main__":
    loader = KnbsDataLoader()
    df = loader.list_reports()
    print("Found reports:")
    print(df[['year', 'month', 'report_type']].value_counts())
    
    print("\n--- Testing Inflation Extraction ---")
    inf = loader.get_latest_inflation()
    if inf:
        print(f"✅ Latest Inflation Rate Found: {inf}%")
    else:
        print("❌ Could not extract inflation rate.")
