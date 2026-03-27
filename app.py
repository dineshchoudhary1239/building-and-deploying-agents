"""
CreditSage AI Advisor — Streamlit Application (Component 3)

UI Features:
- Sidebar: Applicant selector, applicant snapshot panel
- Chat window: Full conversation history
- Intent badge: Shows detected route for each response
- Clear conversation button
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from agent.agent import run_agent
from agent.memory import SessionMemory

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSage AI Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Check API Key ────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "❌ **OPENAI_API_KEY not found.** "
        "Please create a `.env` file with `OPENAI_API_KEY=your_key` and restart the app."
    )
    st.stop()

# ─── Load Dataset ─────────────────────────────────────────────────────────────
@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists("creditsage_loan_applications.csv"):
        st.error("❌ Dataset file `creditsage_loan_applications.csv` not found in project root.")
        st.stop()
    return pd.read_csv("creditsage_loan_applications.csv")

df = load_df()

# ─── Session State Initialisation ─────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "applicant_id" not in st.session_state:
    st.session_state.applicant_id = None
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []     # [(role, content, intent)]
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "prev_applicant" not in st.session_state:
    st.session_state.prev_applicant = None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 CreditSage AI")
    st.caption("Loan Advisory Agent · Set A")
    st.divider()

    # ── Applicant Selector ────────────────────────────────────────────────────
    st.markdown("### 👤 Select Applicant")
    options = ["— Select an applicant —"] + [
        f"{row['applicant_id']}  ·  {row['name']}"
        for _, row in df.iterrows()
    ]
    selected = st.selectbox("Applicant", options, label_visibility="collapsed")

    if selected != "— Select an applicant —":
        new_id = int(selected.split("·")[0].strip())

        # Reset conversation when applicant changes
        if new_id != st.session_state.prev_applicant:
            st.session_state.applicant_id = new_id
            st.session_state.prev_applicant = new_id
            st.session_state.memory.clear()
            st.session_state.chat_display = []
            st.session_state.last_intent = None
            st.success(f"✅ Loaded Applicant #{new_id}")
    else:
        st.session_state.applicant_id = None

    st.divider()

    # ── Applicant Snapshot ────────────────────────────────────────────────────
    if st.session_state.applicant_id:
        row = df[df["applicant_id"] == st.session_state.applicant_id].iloc[0]

        with st.expander("📋 Applicant Snapshot", expanded=True):
            st.markdown(f"**{row['name']}**")
            st.markdown(f"{row['age']} yrs · {row['gender']} · {row['city']}")
            st.markdown(f"🏢 {row['employment_type']} @ {row['employer_name']}")
            st.markdown(f"⏱️ {row['years_at_current_job']} yrs tenure")
            st.divider()

            c1, c2 = st.columns(2)
            c1.metric("Monthly Income", f"₹{row['monthly_income']:,.0f}")
            c2.metric("Credit Score",   int(row['credit_score']))
            c1.metric("Existing EMI",   f"₹{row['existing_emi']:,.0f}")
            c2.metric("Loan Purpose",   row['loan_purpose'])

            st.divider()
            st.metric("Requested Amount",    f"₹{row['requested_amount']:,.0f}")
            st.metric("Preferred Tenure",    f"{row['preferred_tenure_months']} months")
            st.metric("Down Payment",        f"₹{row['down_payment']:,.0f}")
            st.metric("Collateral",          row['collateral'])

    st.divider()

    # ── Clear Button ──────────────────────────────────────────────────────────
    if st.button("🗑️  Clear Conversation", use_container_width=True, type="secondary"):
        st.session_state.memory.clear()
        st.session_state.chat_display = []
        st.session_state.last_intent = None
        st.rerun()

    st.caption("Built with GPT-4o · Router Pattern · 5 Tools")


# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("# 💳 CreditSage AI Loan Advisor")
st.caption("Powered by GPT-4o with Router Pattern | 5 Specialised Tools | Session Memory")

# ── Intent Badge ──────────────────────────────────────────────────────────────
INTENT_CONFIG = {
    "ELIGIBILITY":   ("🟡", "Eligibility Check",   "#fef3c7"),
    "PRODUCT_MATCH": ("🔵", "Product Matching",    "#dbeafe"),
    "EMI_CALC":      ("🟢", "EMI Calculation",     "#dcfce7"),
    "GENERAL":       ("⚪", "General Advisory",    "#f3f4f6"),
}

if st.session_state.last_intent:
    icon, label, colour = INTENT_CONFIG.get(
        st.session_state.last_intent, ("⚪", "General", "#f3f4f6")
    )
    st.markdown(
        f"""<div style="background:{colour};padding:6px 14px;border-radius:20px;
        display:inline-block;font-size:0.85rem;margin-bottom:8px;">
        {icon} Last route: <strong>{label}</strong></div>""",
        unsafe_allow_html=True,
    )

# ── Chat Window ───────────────────────────────────────────────────────────────
if not st.session_state.chat_display:
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
            <div style="font-size:3rem">💳</div>
            <h3 style="color:#475569">Welcome to CreditSage AI Advisor</h3>
            <p>Select an applicant from the sidebar to begin the advisory session.</p>
            <br/>
            <b>Sample questions you can ask:</b><br/><br/>
            <code>Is this applicant eligible for the loan?</code><br/>
            <code>What loan products are available for them?</code><br/>
            <code>Calculate EMI for 3 years vs 5 years vs 7 years</code><br/>
            <code>What is their risk profile?</code><br/>
            <code>What documents are needed for a home loan?</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for role, content, _ in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(content)

# ── Input Box ─────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask about eligibility, loan products, EMI calculations, or anything else…"
)

if user_input:
    if not st.session_state.applicant_id:
        st.warning("⚠️ Please select an applicant from the sidebar first.")
        st.stop()

    # Show user message immediately
    st.session_state.chat_display.append(("user", user_input, None))

    # Run the agent
    with st.spinner("🤔 Analysing your query…"):
        response_text, detected_intent = run_agent(
            user_message=user_input,
            applicant_id=st.session_state.applicant_id,
            conversation_history=st.session_state.memory.get_history(),
        )

    # Update memory (for next turn context)
    st.session_state.memory.add_user(user_input)
    st.session_state.memory.add_assistant(response_text)

    # Store for display
    st.session_state.last_intent = detected_intent
    st.session_state.chat_display.append(("assistant", response_text, detected_intent))

    st.rerun()
