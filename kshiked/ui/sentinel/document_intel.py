"""Document Intelligence tab - Live News + PDF Dossiers."""

from ._shared import st, datetime, Path


def render_document_intel_tab(theme):
    """Render Intelligence: Live News + PDF Dossiers."""
    try:
        from document_intel import get_document_intel
        intel_data = get_document_intel().get_snapshot()
    except Exception as e:
        st.error(f"Intelligence System Offline: {e}")
        return

    # Sub-navigation
    cols = st.columns([1, 1, 4])
    with cols[0]:
        view_mode = st.radio("Source", ["Live News", "Local Dossiers"], label_visibility="collapsed")

    st.markdown("---")

    # ==========================================
    # VIEW: LIVE NEWS (NewsAPI)
    # ==========================================
    if view_mode == "Live News":
        news_data = intel_data.get("news", {})

        if not news_data:
            st.info("No news data available. Check API quota or connectivity.")
            return

        categories = sorted(list(news_data.keys()))
        selected_cat = st.selectbox("Category", [c.upper() for c in categories])

        if selected_cat:
            cat_key = selected_cat.lower()
            articles = news_data.get(cat_key, [])

            st.markdown(f"### {selected_cat} ({len(articles)} Articles)")

            if not articles:
                st.info(f"No recent articles in {selected_cat}.")
            else:
                for art in articles:
                    pub = art.get('published_at', '')
                    time_label = pub
                    try:
                        dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                        now = datetime.now(dt.tzinfo)
                        diff = now - dt
                        if diff.days > 0:
                            time_label = f"{diff.days}d ago"
                        elif diff.seconds > 3600:
                            time_label = f"{diff.seconds // 3600}h ago"
                        else:
                            time_label = f"{diff.seconds // 60}m ago"
                    except Exception:
                        pass

                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 0.8rem; padding: 1rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: {theme.accent_primary}; font-size: 0.8rem; font-weight: 700;">
                                {art.get('source', 'Unknown').upper()}
                            </span>
                            <span style="color: {theme.text_muted}; font-size: 0.8rem;">
                                {time_label}
                            </span>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">
                            <a href="{art.get('url')}" target="_blank" style="text-decoration: none; color: {theme.text_primary};">
                                {art.get('title')}
                            </a>
                        </div>
                        <div style="font-size: 0.9rem; color: {theme.text_secondary};">
                            {art.get('description') or ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    excerpt = art.get("evidence_excerpt") or (art.get("extracted_text") or "")[:260]
                    if excerpt:
                        st.caption(f"Evidence excerpt: {excerpt}")

                    trace_parts = []
                    if art.get("article_id"):
                        trace_parts.append(f"article_id={art.get('article_id')}")
                    if art.get("content_record_id"):
                        trace_parts.append(f"content_id={art.get('content_record_id')}")
                    if art.get("content_storage_path"):
                        trace_parts.append(f"path={art.get('content_storage_path')}")
                    if trace_parts:
                        st.caption(f"Trace: {' | '.join(trace_parts)}")

    # ==========================================
    # VIEW: LOCAL DOSSIERS (PDFs)
    # ==========================================
    elif view_mode == "Local Dossiers":
        if "dossier_nav" not in st.session_state:
            st.session_state.dossier_nav = {"view": "themes", "category": None, "file": None}

        base_dir = Path("random/content_extracted")
        if not base_dir.exists():
            st.info("No extracted dossiers found in random/content_extracted.")
            return

        nav = st.session_state.dossier_nav
        header_text = "LOCAL DOSSIERS"
        if nav["view"] == "list":
            header_text += f" / {nav['category'].replace('_', ' ').upper()}"
        elif nav["view"] == "content":
            header_text += f" / {nav['category'].replace('_', ' ').upper()} / READING"

        st.markdown(f'<div class="section-header">{header_text}</div>', unsafe_allow_html=True)

        # VIEW 1: THEMES (Categories)
        if nav["view"] == "themes":
            categories = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "Uncategorized"])

            if not categories:
                st.info("No dossier themes available.")
                return

            cols = st.columns(3)
            for idx, category in enumerate(categories):
                cat_name = category.replace("_", " ").upper()
                cat_dir = base_dir / category
                file_count = len(list(cat_dir.glob("*.md")))

                with cols[idx % 3]:
                    label = f"{cat_name}\n\n{file_count} Dossiers"
                    if st.button(label, key=f"cat_{category}", use_container_width=True):
                        st.session_state.dossier_nav = {"view": "list", "category": category, "file": None}
                        st.rerun()

        # VIEW 2: ARTICLE LIST
        elif nav["view"] == "list":
            if st.button("<- Back to Themes", key="back_to_themes"):
                st.session_state.dossier_nav = {"view": "themes", "category": None, "file": None}
                st.rerun()

            category = nav["category"]
            st.markdown(f'<div style="margin-bottom: 1rem; color: {theme.accent_info}; font-weight: 600;">THEME: {category.replace("_", " ").upper()}</div>', unsafe_allow_html=True)

            cat_dir = base_dir / category
            files = list(cat_dir.glob("*.md"))

            if not files:
                st.info("No articles found in this theme.")
                return

            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for file_path in files:
                file_name = file_path.stem.replace("_", " ").title()
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                file_size = f"{file_path.stat().st_size / 1024:.1f} KB"

                summary = ""
                try:
                    raw_text = file_path.read_text(encoding="utf-8")[:1000]
                    lines = [l.strip() for l in raw_text.split('\n') if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("|")]
                    if lines:
                        summary = " ".join(lines)[:250] + "..."
                except Exception:
                    summary = "No preview available."

                label = f"{file_name}\n\n{summary}\n\n{file_date} | {file_size}"
                if st.button(label, key=f"read_{file_path.name}", use_container_width=True):
                    st.session_state.dossier_nav = {"view": "content", "category": category, "file": str(file_path)}
                    st.rerun()

        # VIEW 3: ARTICLE CONTENT
        elif nav["view"] == "content":
            file_path = Path(nav["file"])

            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("<- Back", key="back_to_list"):
                    st.session_state.dossier_nav["view"] = "list"
                    st.session_state.dossier_nav["file"] = None
                    st.rerun()

            try:
                content = file_path.read_text(encoding="utf-8")
                st.markdown(f"""
                <div class="glass-card" style="padding: 2rem; border: 1px solid {theme.accent_primary}; margin-top: 1rem;">
                <div style="font-family: 'Courier New', monospace; color: {theme.accent_primary}; font-size: 0.8rem; margin-bottom: 1rem;">
                    SOURCE: {file_path.name}
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(content, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")
