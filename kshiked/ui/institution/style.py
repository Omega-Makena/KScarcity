import streamlit as st
import base64
import os

def get_base64_of_bin_file(bin_file):
  try:
    with open(bin_file, 'rb') as f:
      data = f.read()
    return base64.b64encode(data).decode()
  except Exception as e:
    return ""

@st.cache_data
def inject_enterprise_theme(include_watermark=True):
  """Injects custom CSS to add a clean, highly professional, 
  enterprise-grade control center feel with Kenyan Flag Colors & GOK Background."""
  
  logo_path = r"C:\Users\omegam\OneDrive - Innova Limited\scace4\GOK.png"
  img_b64 = get_base64_of_bin_file(logo_path)
  
  bg_css = ""
  if include_watermark and img_b64:
    bg_css = f"""
    .stApp {{
      background-image: url("data:image/png;base64,{img_b64}") !important;
      background-size: 500px auto !important; /* Large centered watermark */
      background-repeat: no-repeat !important;
      background-position: center center !important;
      background-attachment: fixed !important;
      background-color: #f8fafc !important;
    }}
    [data-testid="stAppViewContainer"] {{
      background-color: rgba(255, 255, 255, 0.92) !important; /* Semi-transparent white to wash out the watermark slightly */
    }}
    """
  else:
    bg_css = """
    .stApp {{
      background-color: #f8fafc !important;
      background-image: none !important;
    }}
    """

  st.markdown(f"""
    <style>
    /* Modern Font Injection */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
      font-family: 'IBM Plex Sans', sans-serif !important;
    }}

    /* Numeric display surfaces use mono for readability and stable alignment. */
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    .kpi-value,
    .mono,
    .stNumberInput input,
    input[type="number"],
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .hovertext {{
      font-family: 'IBM Plex Mono', monospace !important;
      font-variant-numeric: tabular-nums;
    }}

    /* Fluid Animations Keyframes */
    @keyframes slideUpFade {{
      0% {{ opacity: 0; transform: translateY(15px); }}
      100% {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulseGlow {{
      0% {{ box-shadow: 0 0 0 0 rgba(187, 0, 0, 0.4); }}
      70% {{ box-shadow: 0 0 0 10px rgba(187, 0, 0, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(187, 0, 0, 0); }}
    }}

    /* Clean solid backgrounds for Expanders */
    [data-testid="stExpander"] {{
      background: #ffffff !important;
      border: 1px solid #000000 !important; /* Kenyan Black */
      border-radius: 8px !important;
      box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.05) !important;
      margin-bottom: 0.75rem;
      transition: all 0.2s ease-in-out;
      animation: slideUpFade 0.4s ease-out forwards;
    }}

    [data-testid="stExpander"]:hover {{
      border-color: #006600 !important; /* Kenyan Green */
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.05) !important;
      transform: translateY(-1px);
    }}

    /* Expander Headers */
    [data-testid="stExpander"] summary {{
      background-color: #f8fafc !important;
      border-radius: 8px 8px 0 0 !important;
      font-weight: 600 !important;
      color: #000000 !important; /* Kenyan Black */
      padding: 0.75rem 1rem !important;
      border-bottom: 2px solid #BB0000 !important; /* Kenyan Red */
    }}

    /* Clean Glassmorphism Containers (The Cards) */
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background: rgba(255, 255, 255, 0.85) !important;
      backdrop-filter: blur(12px) !important;
      -webkit-backdrop-filter: blur(12px) !important;
      border-radius: 16px !important;
      border: 1px solid rgba(255, 255, 255, 0.4) !important;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01) !important;
      padding: 1.5rem !important;
      animation: slideUpFade 0.5s ease-out forwards;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.02) !important;
      transform: translateY(-2px);
      border: 1px solid rgba(0, 102, 0, 0.2) !important; /* Subtle Kenyan Green on hover */
    }}
    
    /* Premium Custom HTML Components */
    .hero-brief {{
      background: linear-gradient(135deg, rgba(0,0,0,0.95), rgba(0,0,0,0.85));
      color: white;
      padding: 24px;
      border-radius: 12px;
      border-left: 6px solid #BB0000;
      margin-bottom: 20px;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }}
    .hero-brief h3 {{
      color: white !important;
      margin-top: 0;
      font-weight: 600;
    }}
    .hero-brief p {{
      color: #e2e8f0;
      font-size: 1.1rem;
      line-height: 1.6;
    }}
    
    .threat-card {{
      background: white;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 12px;
      border-left: 4px solid #e2e8f0;
      box-shadow: 0 2px 4px rgba(0,0,0,0.02);
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: transform 0.2s;
    }}
    .threat-card:hover {{
      transform: translateX(4px);
      background: #f8fafc;
    }}
    .threat-card.high {{ border-left-color: #BB0000; }}
    .threat-card.medium {{ border-left-color: #F59E0B; }}
    .threat-card.low {{ border-left-color: #3B82F6; }}
    
    .threat-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .threat-title {{
      font-weight: 600;
      color: #0f172a;
      font-size: 1rem;
    }}
    .threat-meta {{
      font-size: 0.8rem;
      color: #64748b;
    }}
    .threat-sector {{
      display: inline-block;
      background: #f1f5f9;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
      color: #475569;
    }}

    /* Input Fields - Flat and Crisp */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {{
      background: #ffffff !important;
      border: 1px solid #000000 !important; /* Kenyan Black */
      border-radius: 6px !important;
      box-shadow: none !important;
      color: #000000 !important;
      transition: all 0.2s ease;
    }}
    
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox [data-baseweb="select"]:focus {{
      background: #ffffff !important;
      border-color: #006600 !important; /* Kenyan Green focus */
      box-shadow: 0 0 0 2px rgba(0, 102, 0, 0.2) !important;
    }}
    
    /* Buttons */
    .stButton button {{
      border-radius: 6px !important;
      font-weight: 600 !important;
      transition: all 0.2s ease;
      border: 2px solid #000000 !important;
    }}
    
    /* Primary Button (Black with Red on Hover) */
    .stButton button[kind="primary"] {{
      background-color: #000000 !important; /* Kenyan Black */
      color: #ffffff !important;
      border: none !important;
    }}
    .stButton button[kind="primary"]:hover {{
      background-color: #BB0000 !important; /* Kenyan Red */
      color: #ffffff !important;
      transform: translateY(-1px);
      box-shadow: 0 4px 6px -1px rgba(187, 0, 0, 0.3) !important;
      border: none !important;
    }}

    /* Secondary Button Hover */
    .stButton button[kind="secondary"]:hover {{
      border-color: #006600 !important; /* Kenyan Green */
      color: #006600 !important;
    }}

    {bg_css}
    
    /* Tabs Premium Minimal Styling and Sticky Navigation */
    [data-testid="stTabs"] > div:first-of-type {{
      position: sticky !important;
      top: 0 !important;
      z-index: 999990 !important;
      background: rgba(248, 250, 252, 0.95) !important;
      backdrop-filter: blur(12px) !important;
      -webkit-backdrop-filter: blur(12px) !important;
      padding-top: 10px !important;
      padding-bottom: 5px !important;
      margin-bottom: 15px !important;
      border-bottom: 1px solid #e2e8f0 !important;
    }}
    
    [data-testid="stTabs"] button {{
      border-bottom: 2px solid transparent !important;
      color: #000000 !important;
      transition: all 0.2s ease;
      font-weight: 500 !important;
      padding-bottom: 0.5rem !important;
    }}
    
    [data-testid="stTabs"] button[aria-selected="true"] {{
      border-bottom: 3px solid #BB0000 !important; /* Kenyan Red */
      color: #BB0000 !important;
      font-weight: 700 !important;
      background: transparent !important;
      border-radius: 0 !important;
    }}
    
    /* Explicit Typography Overrides for Headers */
    h1, h2, h3, h4, h5, h6 {{
      color: #000000 !important; /* Kenyan Black */
      font-weight: 700 !important;
      letter-spacing: -0.025em !important;
    }}
    
    /* Custom KPI Card Classes (Used via markdown) */
    .kpi-card {{
      background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
      border: 1px solid #000000; /* Kenyan Black */
      border-left: 5px solid #006600; /* Kenyan Green */
      border-radius: 8px;
      padding: 1.25rem;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
      animation: slideUpFade 0.4s ease-out forwards;
    }}
    
    .kpi-card.alert {{
      border-left-color: #BB0000; /* Kenyan Red */
      animation: slideUpFade 0.4s ease-out forwards, pulseGlow 2s infinite;
    }}
    
    .kpi-title {{
      color: #000000; /* Kenyan Black */
      font-size: 0.875rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }}
    
    .kpi-value {{
      color: #BB0000; /* Kenyan Red */
      font-size: 2.25rem;
      font-weight: 800;
      line-height: 1.2;
    }}
    
    .kpi-sub {{
      color: #475569;
      font-size: 0.875rem;
      margin-top: 0.25rem;
      font-weight: 500;
    }}
    
    hr {{
      border-color: #000000 !important;
      border-width: 2px !important;
    }}
    </style>
  """, unsafe_allow_html=True)

