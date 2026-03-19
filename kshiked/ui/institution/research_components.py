import streamlit as st
import asyncio
from kshiked.ui.institution.backend.research_engine import ResearchEngine

def render_research_engine_panel(engine: ResearchEngine, theme=None):
  """
  Renders a unified, general-purpose Research Engine workbench.
  """
  
  # Optional theme fallbacks
  bg_card = getattr(theme, 'bg_card', '#FFFFFF')
  border_col = getattr(theme, 'border_default', '#E5E7EB')
  text_primary = getattr(theme, 'text_primary', '#111827')
  text_muted = getattr(theme, 'text_muted', '#6B7280')
  accent_primary = getattr(theme, 'accent_primary', '#2563EB')

  st.markdown(f"""
    <div style="background:{bg_card}; padding:2rem; border-radius:12px; border:1px solid {border_col}; margin-bottom:2rem; text-align:center;">
      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="{accent_primary}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:10px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
      <h2 style="margin-top:0; color:{text_primary}; margin-bottom:0.8rem; font-weight:800; letter-spacing:-0.05em;">K-SHIELD RESEARCH ENGINE</h2>
      <p style="color:{text_muted}; font-size:1.05rem; max-width:800px; margin:0 auto; line-height:1.6;">
        A general-purpose intelligence orchestrator for profound analysis on national resilience, systemic shocks, and contextual economic scenarios.
      </p>
    </div>
  """, unsafe_allow_html=True)

  # Initialize chat history in session state specific to the role/scope
  chat_key = f"research_history_{engine.context.role}"
  if chat_key not in st.session_state:
    st.session_state[chat_key] = []

  # Wrapper for chat history
  st.markdown('<div style="display:flex; flex-direction:column; gap:15px; margin-bottom:20px;">', unsafe_allow_html=True)
  
  for msg in st.session_state[chat_key]:
    if msg["role"] == "user":
      st.markdown(f"""
      <div style="align-self: flex-end; width: 100%; display: flex; justify-content: flex-end;">
        <div style="background:#F3F4F6; padding:12px 18px; border-radius:12px; max-width:80%; text-align:right;">
          <div style="color:#6B7280; font-size:0.75rem; text-transform:uppercase; font-weight:700; margin-bottom:4px;">Research Query</div>
          <div style="color:#111827; font-size:1rem;">{msg["content"]}</div>
        </div>
      </div>
      """, unsafe_allow_html=True)
    else:
      st.markdown(f"""
      <div style="align-self: flex-start; width: 100%;">
        <div style="background:#EEF2FF; border-left:4px solid #4F46E5; padding:20px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
          <div style="color:#4338CA; font-size:0.85rem; text-transform:uppercase; font-weight:800; margin-bottom:10px; display:flex; align-items:center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            Engine Analysis
          </div>
          <div style="color:#1E1B4B; font-size:0.95rem; line-height:1.7;">{msg["content"]}</div>
        </div>
      </div>
      """, unsafe_allow_html=True)
      
  st.markdown('</div>', unsafe_allow_html=True)

  st.markdown("---")

  # Input area
  st.markdown("<h4 style='margin-bottom:0.5rem;'>Synthesize New Intelligence</h4>", unsafe_allow_html=True)
  
  with st.form(key="research_query_form", clear_on_submit=True):
    qc1, qc2 = st.columns([4, 1])
    with qc1:
      query = st.text_input("Enter your research query, hypothesis, or scenario:", 
                 placeholder="e.g. Investigate the downstream secondary variables affected by a severe drought in the northeast...", 
                 label_visibility="collapsed")
    with qc2:
      submit = st.form_submit_button("Analyze Data", use_container_width=True)

    if submit and query:
      st.session_state[chat_key].append({"role": "user", "content": query})
      with st.spinner("Synthesizing profound research analysis..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          result = loop.run_until_complete(engine.research(query, scope=engine.context.role))
          st.session_state[chat_key].append({"role": "engine", "content": result.plain_language_summary})
          st.rerun()
        except Exception as e:
          st.error(f"Engine Error: {str(e)}")
        finally:
          loop.close()
