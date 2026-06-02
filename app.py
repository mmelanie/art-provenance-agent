"""
Art Provenance Research Agent — Streamlit Web App

Run with:
    streamlit run app.py
"""

import json
import os
import sys

import streamlit as st

# Allow importing from parent directory if running from art-provenance-agent folder
sys.path.insert(0, os.path.expanduser("~"))
from art_provenance_agent import research_provenance, ProvenanceLog


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Art Provenance Research Agent",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Art Provenance Research Agent")
st.caption("Powered by Claude AI + Tavily Search")
st.markdown(
    "Enter an artwork below and the agent will search auction records, museum records, "
    "exhibition catalogs, and ownership transfers to build a provenance log."
)

st.divider()

# ---------------------------------------------------------------------------
# API key inputs (sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🔑 API Keys")
    st.markdown("Keys are used only for your session and never stored.")

    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        value=os.environ.get("TAVILY_API_KEY", ""),
    )

    st.markdown("---")
    st.markdown("**Get your keys:**")
    st.markdown("• [Anthropic Console](https://console.anthropic.com)")
    st.markdown("• [Tavily Dashboard](https://app.tavily.com)")

# ---------------------------------------------------------------------------
# Artwork input form
# ---------------------------------------------------------------------------

with st.form("artwork_form"):
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "Artwork Title *",
            placeholder="e.g. Sunflowers",
        )
        artist = st.text_input(
            "Artist *",
            placeholder="e.g. Vincent van Gogh",
        )

    with col2:
        approximate_date = st.text_input(
            "Approximate Date *",
            placeholder="e.g. 1888 or c. 1900–1910",
        )
        st.markdown("")  # spacer
        st.markdown("")  # spacer

    submitted = st.form_submit_button("🔍 Research Provenance", use_container_width=True)

# ---------------------------------------------------------------------------
# Run agent on submission
# ---------------------------------------------------------------------------

if submitted:
    # Validate inputs
    if not title or not artist or not approximate_date:
        st.error("Please fill in all three fields.")
        st.stop()

    if not anthropic_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    if not tavily_key:
        st.error("Please enter your Tavily API key in the sidebar.")
        st.stop()

    # Run the agent with a progress indicator
    with st.spinner(f"Researching provenance for **{title}** by {artist}… this may take a minute."):
        try:
            log: ProvenanceLog = research_provenance(
                title=title,
                artist=artist,
                approximate_date=approximate_date,
                anthropic_api_key=anthropic_key,
                tavily_api_key=tavily_key,
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.success("Research complete!")
    st.divider()

    # ---------------------------------------------------------------------------
    # Display results
    # ---------------------------------------------------------------------------

    st.subheader(f"📋 Provenance Log: *{log.artwork_title}*")
    st.markdown(f"**Artist:** {log.artist}  &nbsp;|&nbsp;  **Date:** {log.approximate_date}")

    if log.summary:
        st.info(log.summary)

    # Ownership chain
    if log.entries:
        st.markdown(f"### ⛓️ Chain of Ownership &nbsp; `{len(log.entries)} records`")
        for i, entry in enumerate(log.entries, 1):
            with st.expander(
                f"{i}. **{entry.owner}** &nbsp; [{entry.from_date} – {entry.to_date}]",
                expanded=True,
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Acquisition method:** {entry.acquisition_method}")
                with col_b:
                    st.markdown(f"**Period:** {entry.from_date} – {entry.to_date}")
                if entry.source:
                    st.markdown(f"**Source:** {entry.source}")
                if entry.notes:
                    st.markdown(f"**Notes:** {entry.notes}")
    else:
        st.warning("No ownership records found.")

    # Gaps
    if log.gaps:
        st.markdown("### ⚠️ Provenance Gaps")
        for gap in log.gaps:
            st.warning(gap)

    # Red flags
    if log.red_flags:
        st.markdown("### 🚩 Red Flags")
        for flag in log.red_flags:
            st.error(flag)

    # Sources
    if log.sources_consulted:
        with st.expander(f"📚 Sources Consulted ({len(log.sources_consulted)})"):
            for src in log.sources_consulted:
                st.markdown(f"• {src}")

    st.divider()

    # Download button
    json_str = json.dumps(
        {
            "artwork_title": log.artwork_title,
            "artist": log.artist,
            "approximate_date": log.approximate_date,
            "summary": log.summary,
            "entries": [
                {
                    "owner": e.owner,
                    "from_date": e.from_date,
                    "to_date": e.to_date,
                    "acquisition_method": e.acquisition_method,
                    "source": e.source,
                    "notes": e.notes,
                }
                for e in log.entries
            ],
            "gaps": log.gaps,
            "red_flags": log.red_flags,
            "sources_consulted": log.sources_consulted,
        },
        indent=2,
        ensure_ascii=False,
    )

    filename = f"provenance_{log.artist.replace(' ', '_')}_{log.artwork_title.replace(' ', '_')}.json"

    st.download_button(
        label="⬇️ Download Provenance Log (JSON)",
        data=json_str,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )
