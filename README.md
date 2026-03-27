# 💳 CreditSage AI Loan Advisor

An AI-powered loan advisory agent built for CreditSage Financial Technologies.
Built with **Python**, **Llama-3.3-70B via Groq**, and **Streamlit** using a **Router Pattern** and **5 specialised tools**.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/dineshchoudhary1239/building-and-deploying-agents.git
cd building-and-deploying-agents
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
```
Open `.env` and add your OpenAI API key:
```
GROQ_API_KEY=gsk_...your_key_here...
```

### 5. Place the Dataset
Ensure `creditsage_loan_applications.csv` is in the **project root** (same folder as `run.py`).

### 6. Run the App
```bash
python run.py
```
The Streamlit app will open at **http://localhost:8501**

---

## 🏗️ Agentic Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│      ROUTER AGENT           │  ← LLM-based intent classification
│  (llama-3.3-70b, temp=0.0)   │    NOT keyword matching
└────────────┬────────────────┘
             │ Intent: ELIGIBILITY / PRODUCT_MATCH / EMI_CALC / GENERAL
             ▼
┌─────────────────────────────┐
│    ADVISORY AGENT           │  ← Tool-calling loop (llama-3.3-70b via Groq)
│    (OpenAI Function Call)   │    Up to 6 iterations
└────────────┬────────────────┘
             │ Calls tools based on intent
    ┌────────┴────────┐
    │   5 TOOLS       │
    ├─────────────────┤
    │ check_eligibility       │  → Age, credit score, income, FOIR rules
    │ get_loan_products       │  → Hardcoded product catalog (3 per category)
    │ calculate_emi           │  → Reducing-balance formula
    │ assess_risk_profile     │  → Scoring model (4 factors, 0–100)
    │ get_applicant_summary   │  → Full CSV row retrieval
    └─────────────────┘
             │
             ▼
    SESSION MEMORY (SessionMemory class)
    Retains last 20 messages within session
```

### Router Pattern
The **Router Agent** uses a separate GPT-4o-mini call at `temperature=0.0` to classify every user query into:
| Intent | Triggered By |
|--------|-------------|
| `ELIGIBILITY` | Questions about qualifying, criteria, rejection reasons |
| `PRODUCT_MATCH` | Best loan product, options, comparison, recommendations |
| `EMI_CALC` | Monthly payments, what-if scenarios, affordability |
| `GENERAL` | Documents, process, general advice, greetings |

---

## 📁 Project Structure

```
├── run.py                              ← Entry point (python run.py)
├── app.py                              ← Streamlit UI
├── agent/
│   ├── __init__.py
│   ├── tools.py                        ← All 5 tool implementations
│   ├── router.py                       ← LLM-based intent router
│   ├── memory.py                       ← Session memory management
│   └── agent.py                        ← Main agent orchestrator
├── creditsage_loan_applications.csv    ← Dataset (place in root)
├── requirements.txt
├── .env.example                        ← Environment variable template
├── .gitignore
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key — free at https://console.groq.com/keys (required) |

See `.env.example` for the template.

---

## 📊 Eligibility Rules

| Rule | Threshold |
|------|-----------|
| Age | 21 – 60 years |
| Credit Score (Personal/Vehicle) | ≥ 650 |
| Credit Score (Home/Business) | ≥ 700 |
| Monthly Income — Personal | ≥ ₹25,000 |
| Monthly Income — Vehicle | ≥ ₹30,000 |
| Monthly Income — Home | ≥ ₹40,000 |
| Monthly Income — Business | ≥ ₹50,000 |
| FOIR (existing EMI / income) | ≤ 55% |

---

## 🧮 EMI Formula

```
EMI = P × r × (1+r)^n / ((1+r)^n – 1)

where:
  P = Principal amount
  r = Annual rate / 12 / 100  (monthly rate as decimal)
  n = Tenure in months
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.30 | Groq API (OpenAI-compatible) + function calling |
| `streamlit` | ≥1.35 | Web UI |
| `pandas` | ≥2.0 | CSV data handling |
| `python-dotenv` | ≥1.0 | Environment variable loading |

---

## ⚠️ Notes

- **No API keys are committed** — use `.env` file only
- Dataset CSV must be in the **project root** (relative path)
- Entire solution runs **100% locally** except for OpenAI API calls
- No cloud deployment, no external databases

---

*Built for CreditSage Financial Technologies · Mid-Term Examination · Set A*
