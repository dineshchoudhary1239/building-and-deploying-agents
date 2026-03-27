"""
CreditSage AI Advisor — Main Agent

Orchestrates the full advisory workflow:
1. Router classifies intent (ELIGIBILITY / PRODUCT_MATCH / EMI_CALC / GENERAL)
2. Agent selects and calls the appropriate tools via OpenAI function calling
3. Final LLM response is grounded in real dataset values

Architecture: Router Pattern + Tool-Calling Agent + Session Memory
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from agent.tools import (
    check_eligibility,
    get_loan_products,
    calculate_emi,
    assess_risk_profile,
    get_applicant_summary,
)
from agent.router import classify_intent

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─── Tool Schemas (OpenAI Function Calling Format) ────────────────────────────
# Each schema precisely describes what the tool does, its parameters, and
# expected types — enabling the LLM to call tools correctly and confidently.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Check whether a loan applicant meets CreditSage's minimum eligibility "
                "criteria: age (21–60), credit score floor (650 for Personal/Vehicle, "
                "700 for Home/Business), minimum monthly income thresholds, and FOIR "
                "(existing EMI / income ≤ 55%). Returns eligible status, reason, and "
                "failed criteria list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant ID (1–25) from the dataset.",
                    }
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_products",
            "description": (
                "Retrieve up to 3 matching loan products for a given loan purpose and "
                "requested amount. Returns products sorted by interest rate (lowest first), "
                "with annual rate, max tenure, processing fees, and a sample EMI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_purpose": {
                        "type": "string",
                        "enum": ["Personal", "Home", "Vehicle", "Business"],
                        "description": "The type of loan the applicant is seeking.",
                    },
                    "requested_amount": {
                        "type": "number",
                        "description": "Requested loan amount in INR.",
                    },
                },
                "required": ["loan_purpose", "requested_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_emi",
            "description": (
                "Calculate the monthly EMI using the standard reducing-balance formula: "
                "EMI = P × r × (1+r)^n / ((1+r)^n – 1), where r = annual_rate/12/100 "
                "and n = tenure_months. Also returns total interest payable and total "
                "amount payable. Supports multi-scenario comparisons (call multiple times "
                "with different tenures or rates)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan principal amount in INR.",
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "Annual interest rate as a percentage (e.g., 10.5 for 10.5%).",
                    },
                    "tenure_months": {
                        "type": "integer",
                        "description": "Repayment tenure in months (e.g., 12, 24, 36, 60).",
                    },
                },
                "required": ["principal", "annual_rate", "tenure_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_risk_profile",
            "description": (
                "Assess an applicant's risk tier — Low, Medium, or High — based on a "
                "composite score (0–100) derived from: credit score (40 pts), FOIR/debt-"
                "to-income ratio (30 pts), employment stability in years (20 pts), and "
                "loan-to-income ratio (10 pts). Returns risk tier, score, and factor breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant ID (1–25) from the dataset.",
                    }
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_applicant_summary",
            "description": (
                "Retrieve all available data for an applicant from the CSV database: "
                "demographics (name, age, city, gender), employment details, financial "
                "profile (income, credit score, existing EMI), loan request details "
                "(purpose, amount, tenure, down payment, collateral), and derived metrics "
                "(annual income, FOIR, loan-to-income ratio, net monthly surplus). "
                "Always call this first when a new applicant is being discussed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "Unique applicant ID (1–25) from the dataset.",
                    }
                },
                "required": ["applicant_id"],
            },
        },
    },
]

# Map tool names to their Python implementations
TOOL_FUNCTIONS = {
    "check_eligibility":    check_eligibility,
    "get_loan_products":    get_loan_products,
    "calculate_emi":        calculate_emi,
    "assess_risk_profile":  assess_risk_profile,
    "get_applicant_summary": get_applicant_summary,
}


# ─── Agent System Prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CreditSage AI Advisor — an intelligent, data-driven loan advisory agent for CreditSage Financial Technologies.

You help salaried professionals, self-employed individuals, and business owners evaluate their loan options, understand eligibility, calculate EMIs, and receive personalised financial guidance.

## Your Tools:
1. **get_applicant_summary**  — Retrieve complete applicant profile from the database
2. **check_eligibility**      — Verify eligibility against CreditSage policy rules
3. **get_loan_products**      — Find matching loan products with rates & fees
4. **calculate_emi**          — Compute EMI for any principal/rate/tenure scenario
5. **assess_risk_profile**    — Determine risk tier (Low / Medium / High)

## Core Rules:
- ALWAYS call tools to retrieve data — NEVER fabricate financial figures
- NEVER quote an income, credit score, EMI, or rate that you have not fetched from a tool
- When an Applicant ID is in context, use it proactively — the user should never repeat it
- For EMI what-if queries, call calculate_emi MULTIPLE TIMES (e.g., 3yr, 5yr, 7yr) and present a comparison table
- For PRODUCT_MATCH queries, always check eligibility alongside product recommendations
- All responses must be grounded in dataset values — cross-check figures before stating them

## Response Formatting:
- Use ₹ symbol for Indian Rupees
- Format amounts with Indian numbering (₹10,00,000)
- Use markdown tables for comparisons
- Use bold headings for sections
- Keep responses concise but complete
- End responses with one helpful follow-up suggestion

## Tone:
Professional, empathetic, and clear. Explain financial concepts in plain language."""


# ─── Main Agent Runner ────────────────────────────────────────────────────────
def run_agent(
    user_message: str,
    applicant_id: int,
    conversation_history: list,
) -> tuple:
    """
    Run the full CreditSage advisory agent for a single user turn.

    Flow:
    1. Router classifies query intent via LLM
    2. Build messages with system prompt + conversation history
    3. Agent calls tools in a loop (max 5 iterations) until final response
    4. Return response text + detected intent

    Args:
        user_message:         The user's current query
        applicant_id:         Currently selected applicant (from sidebar)
        conversation_history: Full chat history in OpenAI format

    Returns:
        (response_text: str, intent: str)
    """

    # Step 1: Route — classify query intent using LLM
    intent_result = classify_intent(user_message, conversation_history)
    intent = intent_result.get("intent", "GENERAL")

    # Step 2: Build the message list
    # Inject applicant context into the system prompt for every turn
    system_content = SYSTEM_PROMPT
    if applicant_id:
        system_content += (
            f"\n\n## Active Session:\nApplicant ID **{applicant_id}** is loaded. "
            f"Use this ID automatically when calling tools unless the user specifies a different ID."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(conversation_history)

    # Tag the message with detected intent to guide the LLM's tool selection
    tagged_message = f"[Detected Intent: {intent}] {user_message}"
    messages.append({"role": "user", "content": tagged_message})

    # Step 3: Agentic tool-calling loop
    max_iterations = 6  # Safety cap to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",     # LLM decides whether/which tools to call
            temperature=0.3,        # Low: factual accuracy over creativity
            top_p=0.9,              # Slight diversity for natural language output
            max_tokens=1800,        # Sufficient for detailed advisory responses
        )

        msg = response.choices[0].message

        # No tool calls → final response ready
        if not msg.tool_calls:
            final_text = msg.content or "I was unable to generate a response. Please try again."
            return final_text, intent

        # Append the assistant's tool-call request to the message history
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            if func_name in TOOL_FUNCTIONS:
                tool_result = TOOL_FUNCTIONS[func_name](**func_args)
            else:
                tool_result = {"error": f"Unknown tool: {func_name}"}

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(tool_result, ensure_ascii=False),
            })

    # Fallback if max iterations reached
    return (
        "I apologise — I was unable to complete the analysis within the allowed steps. "
        "Please try rephrasing your question.",
        intent,
    )
