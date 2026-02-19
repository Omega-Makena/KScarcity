import pandas as pd
import numpy as np

csv_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\API_KEN_DS2_en_csv_v2_14659.csv"

def analyze_data():
    try:
        raw_df = pd.read_csv(csv_path, skiprows=4)
        
        target_indicators = [
            "GDP growth (annual %)",
            "Inflation, consumer prices (annual %)",
            "GDP (current US$)"
        ]
        
        df_filtered = raw_df[raw_df['Indicator Name'].isin(target_indicators)].copy()
        
        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        value_vars = [c for c in df_filtered.columns if c.isdigit()]
        
        df_long = df_filtered.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
        df_pivot = df_long.pivot(index='Year', columns='Indicator Name', values='Value')
        df_pivot.index = df_pivot.index.astype(int)
        
        print("Data loaded. checking for crashes...")
        
        # Look for GDP Growth < 0 or Inflation > 15
        for year, row in df_pivot.iterrows():
            gdp_growth = row.get("GDP growth (annual %)", np.nan)
            inflation = row.get("Inflation, consumer prices (annual %)", np.nan)
            
            if gdp_growth < 2.0 or inflation > 15.0:
                print(f"Year {year}: GDP Growth={gdp_growth:.2f}%, Inflation={inflation:.2f}%")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_data()
