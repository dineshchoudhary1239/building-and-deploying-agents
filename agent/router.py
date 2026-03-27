"""
CreditSage AI Advisor — Router Agent (Component 2)

Implements LLM-based intent classification (NOT keyword matching).
The router uses a dedicated LLM call to understand the semantic meaning
of the user's query and route it to the appropriate handler.

Routes:
- ELIGIBILITY    → check_eligibility + assess_risk_profile
- PRODUCT_MATCH  → get_loan_products + check_eligibility
- EMI_CALC       → calculate_emi (multi-scenario support)
- GENERAL        → LLM knowledge + applicant context (no tool call)
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Using Groq's OpenAI-compatible API — free tier, fast inference
# Groq is explicitly allowed per exam rules alongside OpenAI/Anthropic
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

VALID_INTENTS = ["ELIGIBILITY", "PRODUCT_MATCH", "EMI_CALC", "GENERAL"]

# ─── Router System Prompt ─────────────────────────────────────────────────────
# This prompt uses semantic understanding — NOT keyword lists.
# The LLM must reason about the user's true intent.
ROUTER_SYSTEM_PROMPT = """You are a query intent classifier for CreditSage, a loan advisory system.

Your sole job is to classify the user's query into exactly ONE of the four intents below.
Use semantic understanding — do NOT rely on simple keywords.

INTENTS:
- ELIGIBILITY    → User wants to know if they/someone qualifies, what criteria they meet or fail,
                   minimum requirements, reasons for rejection/approval, credit score adequacy,
                   income sufficiency, age requirements, or FOIR checks.

- PRODUCT_MATCH  → User wants to know which loan product is best for them, what options exist,
                   product comparison, bank recommendations, interest rates available for their
                   profile, or which loan type suits them.

- EMI_CALC       → User wants to know monthly payment amounts, total payable, what-if scenarios
                   (different tenure, different amount, different rate), affordability analysis,
                   or EMI breakdown. Any calculation involving principal/rate/tenure.

- GENERAL        → Everything else: process questions, documents needed, general financial advice,
                   company information, greetings, clarifications, how the system works, etc.

Respond with ONLY a valid JSON object. No explanation, no extra text:
{"intent": "ELIGIBILITY", "confidence": 0.95, "reason": "brief reason"}"""


def classify_intent(user_message: str, conversation_history: list = None) -> dict:
    """
    Use LLM to classify the user query into one of four intent categories.

    This is LLM-based routing — not keyword matching. The model reasons
    about the semantic meaning of the query considering conversation context.

    Args:
        user_message:         The current user query
        conversation_history: Recent conversation for context (optional)

    Returns:
        dict with keys: intent (str), confidence (float), reason (str)
    """
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]

    # Provide last 4 messages as context so the router understands follow-ups
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast Groq model for classification
            messages=messages,
            temperature=0.0,             # Fully deterministic for consistent routing
            top_p=1.0,
            max_tokens=80,               # Classification response is small
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        # Validate and normalise intent
        if result.get("intent") not in VALID_INTENTS:
            result["intent"] = "GENERAL"

        return result

    except Exception as e:
        # Fallback to GENERAL on any error — never crash the app
        return {
            "intent": "GENERAL",
            "confidence": 0.5,
            "reason": f"Fallback due to classification error: {str(e)}",
        }
