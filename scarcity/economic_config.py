"""
economic configuration.

defines the whitelist of variables for the economic knowledge graph.
focuses on macro-economic indicators: gdp, inflation, fiscal, trade, and resilience.
"""

from typing import Dict

# visual mapping: friendly_name -> indicator_code
ECONOMIC_VARIABLES: Dict[str, str] = {
    # --- growth ---
    "gdp_usd": "NY.GDP.MKTP.CD",               # gdp (current us$)
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",         # gdp growth (annual %)
    "gdp_per_capita": "NY.GDP.PCAP.KD",        # gdp per capita (constant 2015 us$)
    
    # --- prices & inflation ---
    "inflation_cpi": "FP.CPI.TOTL.ZG",         # inflation, consumer prices (annual %)
    "inflation_deflator": "NY.GDP.DEFL.KD.ZG", # inflation, gdp deflator (annual %)
    "real_interest_rate": "FR.INR.RINR",       # real interest rate (%)
    
    # --- fiscal (government) ---
    "gov_expense_gdp": "GC.XPN.TOTL.GD.ZS",    # expense (% of gdp)
    "gov_debt_gdp": "GC.DOD.TOTL.GD.ZS",       # central government debt, total (% of gdp)
    "tax_revenue_gdp": "GC.TAX.TOTL.GD.ZS",    # tax revenue (% of gdp)
    "military_exp_gdp": "MS.MIL.XPND.GD.ZS",   # military expenditure (% of gdp)
    
    # --- trade (external) ---
    "trade_gdp": "NE.TRD.GNFS.ZS",             # trade (% of gdp)
    "current_account": "BN.CAB.XOKA.GD.ZS",    # current account balance (% of gdp)
    "fdi_inflows": "BX.KLT.DINV.WD.GD.ZS",     # foreign direct investment, net inflows (% of gdp)
    
    # --- monetary ---
    "money_broad_gdp": "FM.LBL.BMNY.GD.ZS",    # broad money (% of gdp)
    "dom_credit_pvt": "FS.AST.PRVT.GD.ZS",     # domestic credit to private sector (% of gdp)
    
    # --- labor & social ---
    "unemployment": "SL.UEM.TOTL.ZS",          # unemployment, total (% of total labor force)
    "labor_force_part": "SL.TLF.CACT.ZS",      # labor force participation rate
    "urban_pop_growth": "SP.URB.GROW",         # urban population growth (annual %)
}

# reverse mapping for lookups
CODE_TO_NAME = {v: k for k, v in ECONOMIC_VARIABLES.items()}

def get_friendly_name(code: str) -> str:
    return CODE_TO_NAME.get(code, code)
