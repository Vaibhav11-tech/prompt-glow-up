"""Streamlit frontend for Prompt Glow-Up — Prompt Roaster, Grader, and Rewriter."""

from __future__ import annotations

import html
import os

from dotenv import load_dotenv
import streamlit as st

from engine import (
    DEFAULT_MODEL,
    ComparisonResult,
    InvalidAPIKeyError,
    PromptAudit,
    PromptEngineError,
    RateLimitError,
    RoastIntensity,
    ServiceUnavailableError,
    audit_prompt,
    compare_live_outputs,
)

# Load .env before inspecting environment variables
load_dotenv()

INTENSITY_OPTIONS: list[RoastIntensity] = [
    "Polite Tutor",
    "Honest Reviewer",
    "Brutally Honest Senior Dev",
]

INTENSITY_BADGE_COLORS: dict[RoastIntensity, str] = {
    "Polite Tutor": "#38bdf8",
    "Honest Reviewer": "#f59e0b",
    "Brutally Honest Senior Dev": "#f43f5e",
}


# ---------------------------------------------------------------------------
# Page Setup & Custom Dark Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prompt Glow-Up",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_custom_css() -> None:
    """Inject minimal, high-contrast modern dark CSS."""
    st.markdown(
        """
        <style>
        /* Core dark palette */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] {
            background-color: #0f1624;
            border-right: 1px solid #1e293b;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* Header typography */
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 55%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.02rem;
            margin-bottom: 1.5rem;
        }

        /* Overall score banner */
        .overall-card {
            background: linear-gradient(135deg, #131c31 0%, #172038 100%);
            border: 1px solid #2d3e5e;
            border-radius: 14px;
            padding: 1.3rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        .overall-score-num {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1;
        }
        .overall-tier {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-top: 0.35rem;
        }

        /* Sub-score metric cards */
        .metric-card {
            background-color: #131b2c;
            border: 1px solid #24324a;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            height: 100%;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0.25rem 0 0.55rem 0;
        }
        .progress-track {
            width: 100%;
            height: 7px;
            background-color: #1e293b;
            border-radius: 999px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 999px;
        }

        /* Roast callout */
        .roast-card {
            background-color: #131b2c;
            border: 1px solid #24324a;
            border-left-width: 4px;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin: 1rem 0 1.25rem 0;
        }
        .roast-tag {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.45rem;
        }
        .roast-text {
            color: #f1f5f9;
            font-size: 1.05rem;
            line-height: 1.6;
            margin: 0;
        }

        /* Section labels */
        .section-kicker {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        /* Comparison column headers */
        .compare-badge-before {
            display: inline-block;
            background-color: rgba(244, 63, 94, 0.14);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.35);
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .compare-badge-after {
            display: inline-block;
            background-color: rgba(34, 197, 94, 0.14);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.35);
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }

        /* Textarea polish */
        textarea {
            background-color: #111827 !important;
            border: 1px solid #26354d !important;
            color: #f8fafc !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State & Helpers
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize persistent session keys so reruns never wipe audit results."""
    defaults = {
        "prompt_input": "",
        "audit_result": None,
        "audited_prompt": "",
        "audited_intensity": "Honest Reviewer",
        "comparison_result": None,
        "sidebar_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def score_color(score: int) -> str:
    """Map a 1–10 score to a high-contrast semantic color."""
    if score <= 3:
        return "#f43f5e"  # Rose / red
    if score <= 6:
        return "#f59e0b"  # Amber
    if score <= 8:
        return "#38bdf8"  # Sky blue
    return "#22c55e"      # Emerald


def score_tier_label(score: int) -> str:
    """Return a concise grade tier aligned with the calibrated rubric."""
    if score <= 3:
        return "Needs Life Support"
    if score <= 6:
        return "Half-Baked Draft"
    if score <= 8:
        return "Solid Structure"
    return "Production Airtight"


def resolve_api_key() -> str:
    """Render sidebar config and resolve GEMINI_API_KEY from env or fallback input."""
    env_key = os.getenv("GEMINI_API_KEY", "").strip()

    with st.sidebar:
        st.markdown("### ⚡ Prompt Glow-Up")
        st.caption(f"Model: `{DEFAULT_MODEL}`")
        st.divider()

        st.markdown("#### API Key")
        if env_key:
            st.success("`GEMINI_API_KEY` loaded from environment.", icon="✅")
            active_key = env_key
        else:
            st.warning("No `GEMINI_API_KEY` detected in `.env` or environment.", icon="🔑")
            active_key = st.text_input(
                "Enter Gemini API Key",
                type="password",
                key="sidebar_api_key",
                placeholder="AIza...",
                help="Stored only in your current browser session.",
            ).strip()

        st.divider()
        st.markdown("#### Grading Anchors")
        st.markdown(
            """
            - **1–3:** Vague one-liner
            - **4–6:** Clear intent, missing framing
            - **7–8:** Solid structure, minor gaps
            - **9–10:** Airtight production prompt
            """
        )

    return active_key


# ---------------------------------------------------------------------------
# UI Renderers
# ---------------------------------------------------------------------------
def render_scorecard(audit: PromptAudit) -> None:
    """Render the overall score banner and 4 sub-score metric cards."""
    overall_color = score_color(audit.overall_score)
    tier_label = score_tier_label(audit.overall_score)

    st.markdown(
        f"""
        <div class="overall-card">
            <div>
                <div class="section-kicker" style="margin-bottom: 0.2rem;">Overall Prompt Score</div>
                <div style="color: #cbd5e1; font-size: 0.95rem;">
                    Graded across Clarity, Context, Constraints, and Output Format
                </div>
                <span class="overall-tier" style="background: {overall_color}22; color: {overall_color}; border: 1px solid {overall_color}55;">
                    {html.escape(tier_label)}
                </span>
            </div>
            <div style="text-align: right;">
                <span class="overall-score-num" style="color: {overall_color};">{audit.overall_score}</span>
                <span style="font-size: 1.25rem; color: #64748b; font-weight: 700;">/10</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = [
        ("Clarity", audit.clarity_score),
        ("Context", audit.context_score),
        ("Constraints", audit.constraints_score),
        ("Output Format", audit.format_score),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        color = score_color(value)
        width_pct = value * 10
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">
                        {value}<span style="font-size: 0.95rem; color: #64748b; font-weight: 600;">/10</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {width_pct}%; background-color: {color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_audit_dashboard(
    audit: PromptAudit,
    intensity: RoastIntensity,
    api_key: str,
) -> None:
    """Display the full scorecard, roast, gaps, rewritten prompt, and comparison trigger."""
    st.divider()
    render_scorecard(audit)

    # The Verdict / Roast
    accent = INTENSITY_BADGE_COLORS.get(intensity, "#38bdf8")
    st.markdown(
        f"""
        <div class="roast-card" style="border-left-color: {accent};">
            <div class="roast-tag" style="color: {accent};">
                The Verdict · {html.escape(intensity)}
            </div>
            <p class="roast-text">{html.escape(audit.roast)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weaknesses & Improvements side-by-side
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown("#### 🔍 What Was Missing")
        for item in audit.weaknesses:
            st.markdown(f"- {item}")

    with right_col:
        st.markdown("#### 🚀 Key Upgrades Made")
        for item in audit.improvements_summary:
            st.markdown(f"- {item}")

    st.markdown("")
    st.markdown("#### ✨ The Upgraded Prompt")
    st.caption("Built on Role + Task + Context + Constraints + Output Format. Click the copy icon in the top-right corner.")
    st.code(audit.upgraded_prompt, language="markdown", wrap_lines=True)

    # Bonus Feature: Test Before vs. After
    st.markdown("")
    compare_clicked = st.button(
        "⚔️ Compare Live Outputs",
        type="secondary",
        help="Send both the original prompt and the upgraded prompt to Gemini side-by-side.",
    )

    if compare_clicked:
        if not api_key:
            st.error("Add your Gemini API key in `.env` or the sidebar before running a live comparison.")
        else:
            with st.spinner("Running both prompts live against Gemini..."):
                try:
                    st.session_state["comparison_result"] = compare_live_outputs(
                        original_prompt=st.session_state["audited_prompt"],
                        upgraded_prompt=audit.upgraded_prompt,
                        api_key=api_key,
                    )
                except InvalidAPIKeyError as exc:
                    st.error(f"🔑 {exc}")
                except RateLimitError as exc:
                    st.warning(f"⏳ {exc}")
                except ServiceUnavailableError as exc:
                    st.warning(f"🌩️ {exc}")
                except PromptEngineError as exc:
                    st.error(f"⚠️ {exc}")

    comparison: ComparisonResult | None = st.session_state.get("comparison_result")
    if comparison is not None:
        render_comparison(
            original_prompt=st.session_state["audited_prompt"],
            upgraded_prompt=audit.upgraded_prompt,
            comparison=comparison,
        )


def render_comparison(
    original_prompt: str,
    upgraded_prompt: str,
    comparison: ComparisonResult,
) -> None:
    """Render side-by-side live model outputs for Before vs. After."""
    st.divider()
    st.markdown("### ⚔️ Live Output Comparison")
    st.caption("Same model, same settings—the only difference is the prompt.")

    col_before, col_after = st.columns(2, gap="medium")

    with col_before:
        st.markdown('<span class="compare-badge-before">BEFORE · Original Prompt</span>', unsafe_allow_html=True)
        with st.expander("View original prompt", expanded=False):
            st.code(original_prompt, language="text", wrap_lines=True)
        with st.container(border=True):
            st.markdown(comparison.before_output)

    with col_after:
        st.markdown('<span class="compare-badge-after">AFTER · Upgraded Prompt</span>', unsafe_allow_html=True)
        with st.expander("View upgraded prompt", expanded=False):
            st.code(upgraded_prompt, language="markdown", wrap_lines=True)
        with st.container(border=True):
            st.markdown(comparison.after_output)


# ---------------------------------------------------------------------------
# Main App Flow
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()
    init_session_state()
    api_key = resolve_api_key()

    st.markdown('<div class="hero-title">Prompt Glow-Up</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Paste your laziest prompt. Get a calibrated score, a brutally honest roast, and an engineered rewrite you can benchmark live.</div>',
        unsafe_allow_html=True,
    )

    # Prompt input box
    st.markdown('<div class="section-kicker">Your Rough Prompt</div>', unsafe_allow_html=True)
    st.text_area(
        "Your Rough Prompt",
        key="prompt_input",
        height=160,
        placeholder="Paste a rough or lazy prompt here...",
        label_visibility="collapsed",
    )

    # Roast intensity selector + primary CTA
    controls_col, action_col = st.columns([3, 1.3], vertical_alignment="bottom")
    with controls_col:
        selected_intensity = st.segmented_control(
            "Roast Intensity",
            options=INTENSITY_OPTIONS,
            default="Honest Reviewer",
            selection_mode="single",
        )
        intensity: RoastIntensity = selected_intensity or "Honest Reviewer"

    with action_col:
        glow_up_clicked = st.button(
            "⚡ Glow Up My Prompt",
            type="primary",
            use_container_width=True,
        )

    if glow_up_clicked:
        raw_prompt = st.session_state["prompt_input"].strip()
        if not raw_prompt:
            st.warning("Drop a prompt in the box first before hitting Glow Up.")
        elif not api_key:
            st.error("Missing Gemini API key. Add `GEMINI_API_KEY` to `.env` or paste it in the sidebar.")
        else:
            with st.spinner(f"Auditing your prompt ({intensity} mode)..."):
                try:
                    result = audit_prompt(
                        raw_prompt=raw_prompt,
                        intensity=intensity,
                        api_key=api_key,
                    )
                    st.session_state["audit_result"] = result
                    st.session_state["audited_prompt"] = raw_prompt
                    st.session_state["audited_intensity"] = intensity
                    st.session_state["comparison_result"] = None
                except InvalidAPIKeyError as exc:
                    st.error(f"🔑 {exc}")
                except RateLimitError as exc:
                    st.warning(f"⏳ {exc}")
                except ServiceUnavailableError as exc:
                    st.warning(f"🌩️ {exc}")
                except PromptEngineError as exc:
                    st.error(f"⚠️ {exc}")

    # Persist scorecard & live comparison across reruns
    latest_audit: PromptAudit | None = st.session_state.get("audit_result")
    if latest_audit is not None:
        render_audit_dashboard(
            audit=latest_audit,
            intensity=st.session_state["audited_intensity"],
            api_key=api_key,
        )


if __name__ == "__main__":
    main()