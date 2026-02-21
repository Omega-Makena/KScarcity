import streamlit as st

def inject_enterprise_theme():
    """Injects custom CSS to add a clean, highly professional, 
    enterprise-grade control center feel (flat, high contrast, no gimmicks)."""
    st.markdown("""
        <style>
        /* Clean solid backgrounds for Expanders */
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 4px !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            margin-bottom: 0.75rem;
            transition: border-color 0.2s;
        }

        [data-testid="stExpander"]:hover {
            border-color: #d1d5db !important;
            box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.05) !important;
        }

        /* Expander Headers */
        [data-testid="stExpander"] summary {
            background-color: #f9fafb !important;
            border-radius: 4px 4px 0 0 !important;
            font-weight: 500 !important;
            color: #111827 !important;
        }

        /* Clean Container Borders */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border-radius: 6px !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
            padding: 1.25rem;
        }

        /* Input Fields - Flat and Crisp */
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
            border-radius: 4px !important;
            box-shadow: none !important;
            color: #111827 !important;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox [data-baseweb="select"]:focus {
            border-color: #2563eb !important;
            box-shadow: inset 0 0 0 1px #2563eb !important;
        }
        
        /* Main UI Background layering - Solid Neutral Gray */
        .stApp {
            background-color: #f3f4f6 !important;
            background-image: none !important;
        }
        
        /* Tabs Premium Minimal Styling */
        [data-testid="stTabs"] button {
            border-bottom: 2px solid transparent !important;
            color: #4b5563 !important;
            transition: all 0.2s ease;
        }
        
        [data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 2px solid #111827 !important;
            color: #111827 !important;
            font-weight: 600 !important;
            background: transparent !important;
            border-radius: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
