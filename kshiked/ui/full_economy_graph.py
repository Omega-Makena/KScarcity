"""
Full Economy Knowledge Graph (SFC Model)
Defines the base structure of the macroeconomic system.
"""

def get_macro_graph():
    """Returns (nodes, links) for the full economic map."""
    
    # 1. Define Nodes (ID, Group, Base Value)
    # Groups: Real, Monetary, External, Fiscal, Social
    
    nodes_config = [
        # Real Sector
        ("GDP", "Real"), ("Consumption", "Real"), ("Investment", "Real"), 
        ("Production", "Real"), ("Inventory", "Real"), ("Capital Stock", "Real"),
        
        # Labor & Prices
        ("Wages", "Labor"), ("Employment", "Labor"), ("Unemployment", "Labor"),
        ("Labor Supply", "Labor"), ("Productivity", "Labor"),
        ("Prices", "Price"), ("Inflation", "Price"), ("Energy Costs", "Price"),
        
        # Fiscal
        ("Gov Spending", "Fiscal"), ("Taxes", "Fiscal"), ("Public Debt", "Fiscal"),
        ("Deficit", "Fiscal"),
        
        # Monetary
        ("Money Supply", "Monetary"), ("Interest Rate", "Monetary"), 
        ("Credit Supply", "Monetary"), ("Corporate Debt", "Monetary"),
        ("Household Debt", "Monetary"), ("Bank Reserves", "Monetary"),
        
        # External
        ("Exports", "External"), ("Imports", "External"), ("FX Rate", "External"),
        ("Foreign Reserves", "External"), ("Remittances", "External"),
        
        # Social
        ("Inequality", "Social"), ("Social Stress", "Social"), ("Trust", "Social"),
        ("Political Stability", "Social")
    ]
    
    nodes = [{"id": n, "group": g, "val": 1} for n, g in nodes_config]
    
    # 2. Define Links (Source, Target, Strength 0.1-1.0)
    # Theoretical causal links
    links_config = [
        # Aggregate Demand
        ("Consumption", "GDP", 0.9), ("Investment", "GDP", 0.8), ("Gov Spending", "GDP", 0.7),
        ("Exports", "GDP", 0.6), ("Imports", "GDP", -0.6),
        
        # Consumption loop
        ("GDP", "Wages", 0.8), ("Wages", "Consumption", 0.9), ("Taxes", "Consumption", -0.5),
        ("Household Debt", "Consumption", 0.3),
        
        # Investment loop
        ("GDP", "Investment", 0.6), ("Interest Rate", "Investment", -0.7),
        ("Corporate Debt", "Investment", 0.4), ("Trust", "Investment", 0.5),
        
        # Labor Market
        ("GDP", "Employment", 0.8), ("Employment", "Unemployment", -1.0),
        ("Unemployment", "Wages", -0.6), ("Productivity", "Wages", 0.7),
        
        # Prices
        ("Wages", "Prices", 0.5), ("Energy Costs", "Prices", 0.6),
        ("Money Supply", "Prices", 0.4), ("FX Rate", "Prices", -0.5), # Deprec increases prices
        
        # Fiscal
        ("GDP", "Taxes", 0.8), ("Gov Spending", "Deficit", 1.0), ("Taxes", "Deficit", -1.0),
        ("Deficit", "Public Debt", 1.0), ("Public Debt", "Interest Rate", 0.3),
        
        # Monetary
        ("Central Bank", "Interest Rate", 1.0), ("Interest Rate", "Credit Supply", -0.6),
        ("Credit Supply", "Money Supply", 0.9),
        
        # External
        ("FX Rate", "Exports", 0.7), ("FX Rate", "Imports", -0.6),
        ("GDP", "Imports", 0.8), ("Exports", "Foreign Reserves", 1.0),
        
        # Social Feedback
        ("Inflation", "Social Stress", 0.7), ("Unemployment", "Social Stress", 0.8),
        ("Inequality", "Social Stress", 0.6), ("Social Stress", "Trust", -0.8),
        ("Social Stress", "Political Stability", -0.9),
        ("Political Stability", "Investment", 0.7), ("Political Stability", "FX Rate", 0.5)
    ]
    
    links = []
    for src, tgt, str_ in links_config:
        links.append({
            "source": src, "target": tgt, 
            "value": abs(str_), 
            "color": "#4b5563" if str_ > 0 else "#ef4444", # Grey for pos, Red-ish for neg? Or just grey for structure
            "width": 0.5
        })
        
    return nodes, links
