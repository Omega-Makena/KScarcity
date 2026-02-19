"""Sidebar analysis controls for live signal analysis."""

from ._shared import st


def render_analysis_controls():
    """Sidebar controls for live signal analysis."""
    with st.sidebar:
        st.markdown("### Signal Analysis")

        test_text = st.text_area("Test text:", height=80, key="analysis_input",
                                 placeholder="Enter text to analyze for threat signals...")

        if st.button("Analyze", key="analyze_btn"):
            if test_text:
                try:
                    from kshiked.pulse.sensor import PulseSensor
                    sensor = PulseSensor()
                    detections = sensor.process_text(test_text)
                    if detections:
                        st.success(f"Detected {len(detections)} signals:")
                        for d in detections:
                            name = d.signal_id.name.replace("_", " ").title()
                            st.write(f"* {name}: {d.intensity:.0%}")
                    else:
                        st.info("No signals detected.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
            else:
                st.warning("Enter text to analyze.")

        st.markdown("---")
        st.markdown("### System Info")
        st.caption("SENTINEL v2.0 * KShield Engine")
