"""Streamlit UI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.inference.predict import predict

st.set_page_config(
    page_title="URL Shield - Phishing Detector",
    page_icon="🛡️",
    layout="centered",
)

st.markdown("""
<style>
footer { visibility: hidden; }

.shield-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a3a6b 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.shield-header h1 { color: white !important; margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px; }
.shield-header p  { color: #94b8e0; margin: 0.4rem 0 0; font-size: 0.95rem; }

[data-testid="stForm"] {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.5rem 1.75rem 1.75rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid rgba(128,128,128,0.2);
}

[data-testid="stFormSubmitButton"] {
    margin-top: 0.75rem;
}

[data-testid="stFormSubmitButton"] button {
    background-color: #1a3a6b !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    border: none !important;
    transition: background 0.2s;
}
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #0f1f3d !important;
    border: none !important;
}

.result-safe {
    background: rgba(5, 150, 105, 0.12);
    border-left: 5px solid #059669;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.result-danger {
    background: rgba(220, 38, 38, 0.1);
    border-left: 5px solid #dc2626;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.result-title { font-size: 1.35rem; font-weight: 700; margin: 0; color: var(--text-color); }
.result-sub   { font-size: 0.88rem; margin: 0.35rem 0 0; color: var(--text-color); opacity: 0.65; }

.score-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.6;
    margin: 1rem 0 0.25rem;
}

.feat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(128,128,128,0.15);
    font-size: 0.88rem;
}
.feat-key { color: var(--text-color); opacity: 0.65; }
.feat-val { font-weight: 600; color: var(--text-color); }

.app-footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.45;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.2);
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="shield-header">
  <h1>🛡️ URL Shield</h1>
  <p>Real-time phishing URL detection, powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────
with st.form("analyze_form"):
    url = st.text_input("URL to analyze:", placeholder="https://example.com")
    submitted = st.form_submit_button("Analyze URL")

# ── Result ───────────────────────────────────────────────────────────────────
if submitted and url:
    try:
        result = predict(url)
        prob = result["malicious_probability"]
        is_malicious = result["label"] == 1

        if is_malicious:
            st.markdown("""
            <div class="result-danger">
              <p class="result-title">⚠️ Likely Malicious</p>
              <p class="result-sub">This URL shows characteristics commonly associated with phishing or malware.
              Do not visit it or enter any credentials.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
              <p class="result-title">✅ Likely Safe</p>
              <p class="result-sub">No significant phishing indicators detected.
              Always stay cautious with unfamiliar links.</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score",  f"{prob:.1%}")
        col2.metric("Confidence",  f"{max(prob, 1 - prob):.1%}")
        col3.metric("Verdict",     "Malicious" if is_malicious else "Safe")

        st.markdown('<p class="score-label">Risk level</p>', unsafe_allow_html=True)
        st.progress(prob)

        with st.expander("Feature breakdown"):
            rows = "".join(
                f'<div class="feat-row">'
                f'<span class="feat-key">{k.replace("_", " ").title()}</span>'
                f'<span class="feat-val">{round(v, 4) if isinstance(v, float) else v}</span>'
                f'</div>'
                for k, v in result["features"].items()
            )
            st.markdown(rows, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not load model — train one first. ({e})")
