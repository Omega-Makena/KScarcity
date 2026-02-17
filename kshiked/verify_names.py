"""
Verify indicator names in Kenya dataset.
"""
from pathlib import Path
import pandas as pd

codes = [
    "FR.INR.RINR", 
    "GC.XPN.TOTL.GD.ZS"
]

# Data path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
csv_path = PROJECT_ROOT / "API_KEN_DS2_en_csv_v2_14659.csv"

df = pd.read_csv(csv_path, skiprows=4)
for code in codes:
    match = df[df['Indicator Code'] == code]['Indicator Name'].values
    if len(match) > 0:
        print(match[0])
