"""
CreditSage AI Advisor — Tool-Calling Layer (Component 1)

All five tools the LLM can invoke:
1. check_eligibility      — Age, credit score, income, FOIR rules
2. get_loan_products      — Matching loan product catalog lookup
3. calculate_emi          — Reducing-balance EMI formula
4. assess_risk_profile    — Risk tier scoring (Low / Medium / High)
5. get_applicant_summary  — Full applicant data retrieval from CSV
"""

import os
import pandas as pd

# ─── Dataset Path ─────────────────────────────────────────────────────────────
# Always relative — works regardless of where the project is cloned
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "creditsage_loan_applications.csv")


def _load_data() -> pd.DataFrame:
    """Load the loan applications CSV."""
    return pd.read_csv(CSV_PATH)


# ─── Eligibility Policy Rules ─────────────────────────────────────────────────
# Minimum credit scores per loan type (CreditSage compliance policy)
CREDIT_SCORE_MIN = {
    "personal":  650,   # lower threshold for unsecured personal loans
    "vehicle":   650,   # vehicle acts as implicit collateral
    "home":      700,   # higher bar due to large loan sizes
    "business":  700,   # higher bar due to repayment risk
}

# Minimum monthly income per loan type (INR) — set by risk team
INCOME_MIN = {
    "personal":  25_000,
    "vehicle":   30_000,
    "home":      40_000,
    "business":  50_000,
}

# Maximum Fixed Obligation to Income Ratio (FOIR)
# Existing EMI / Monthly Income must not exceed 55%
MAX_FOIR = 0.55

# ─── Loan Products Catalog ────────────────────────────────────────────────────
# Hardcoded catalog — 3 products per category
# Fields: id, name, rate (annual % ), max_tenure (months), processing_fee_pct,
#         min_amount, max_amount (INR)
LOAN_PRODUCTS = {
    "personal": [
        {
            "id": "PL001", "name": "QuickCash Personal Loan",
            "rate": 13.5, "max_tenure": 60, "processing_fee_pct": 1.5,
            "min_amount": 50_000, "max_amount": 2_000_000,
        },
        {
            "id": "PL002", "name": "FlexPay Personal Loan",
            "rate": 15.0, "max_tenure": 48, "processing_fee_pct": 1.0,
            "min_amount": 50_000, "max_amount": 1_500_000,
        },
        {
            "id": "PL003", "name": "SmartFund Personal",
            "rate": 12.0, "max_tenure": 72, "processing_fee_pct": 2.0,
            "min_amount": 100_000, "max_amount": 3_000_000,
        },
    ],
    "home": [
        {
            "id": "HL001", "name": "DreamHome Loan",
            "rate": 8.5, "max_tenure": 360, "processing_fee_pct": 0.5,
            "min_amount": 500_000, "max_amount": 50_000_000,
        },
        {
            "id": "HL002", "name": "NestEasy Home Loan",
            "rate": 9.0, "max_tenure": 300, "processing_fee_pct": 0.75,
            "min_amount": 500_000, "max_amount": 30_000_000,
        },
        {
            "id": "HL003", "name": "PropertyFirst Loan",
            "rate": 8.75, "max_tenure": 240, "processing_fee_pct": 1.0,
            "min_amount": 1_000_000, "max_amount": 20_000_000,
        },
    ],
    "vehicle": [
        {
            "id": "VL001", "name": "AutoDrive Loan",
            "rate": 10.5, "max_tenure": 84, "processing_fee_pct": 1.0,
            "min_amount": 100_000, "max_amount": 5_000_000,
        },
        {
            "id": "VL002", "name": "WheelEasy Vehicle Loan",
            "rate": 11.0, "max_tenure": 60, "processing_fee_pct": 1.25,
            "min_amount": 50_000, "max_amount": 3_000_000,
        },
        {
            "id": "VL003", "name": "DriveNow Loan",
            "rate": 9.75, "max_tenure": 72, "processing_fee_pct": 0.75,
            "min_amount": 100_000, "max_amount": 4_000_000,
        },
    ],
    "business": [
        {
            "id": "BL001", "name": "BizGrow Business Loan",
            "rate": 14.0, "max_tenure": 60, "processing_fee_pct": 2.0,
            "min_amount": 200_000, "max_amount": 10_000_000,
        },
        {
            "id": "BL002", "name": "EntrepreneurFund Loan",
            "rate": 16.0, "max_tenure": 48, "processing_fee_pct": 1.5,
            "min_amount": 100_000, "max_amount": 5_000_000,
        },
        {
            "id": "BL003", "name": "SME Capital Loan",
            "rate": 12.5, "max_tenure": 84, "processing_fee_pct": 1.75,
            "min_amount": 500_000, "max_amount": 20_000_000,
        },
    ],
}


# ─── Tool 1: check_eligibility ────────────────────────────────────────────────
def check_eligibility(applicant_id: int) -> dict:
    """
    Check whether an applicant meets CreditSage's minimum eligibility criteria.

    Rules enforced:
    - Age must be between 21 and 60 (inclusive)
    - Credit score must meet loan-type minimum (650 for Personal/Vehicle, 700 for Home/Business)
    - Monthly income must meet loan-type minimum
    - FOIR (existing_emi / monthly_income) must not exceed 55%

    Returns:
        dict with keys: eligible (bool), applicant_name, loan_purpose,
                        reason (str), failed_criteria (list)
    """
    df = _load_data()
    row = df[df["applicant_id"] == applicant_id]

    if row.empty:
        return {
            "eligible": False,
            "reason": f"Applicant ID {applicant_id} not found in the dataset.",
            "failed_criteria": ["applicant_not_found"],
        }

    r = row.iloc[0]
    failed = []
    reasons = []

    # Rule 1: Age 21–60
    if not (21 <= r["age"] <= 60):
        failed.append("age")
        reasons.append(
            f"Age {r['age']} is outside the allowed range of 21–60 years."
        )

    # Rule 2: Credit score minimum per loan type
    purpose = r["loan_purpose"].lower()
    min_score = CREDIT_SCORE_MIN.get(purpose, 650)
    if r["credit_score"] < min_score:
        failed.append("credit_score")
        reasons.append(
            f"Credit score {r['credit_score']} is below the minimum "
            f"of {min_score} required for {r['loan_purpose']} loans."
        )

    # Rule 3: Minimum monthly income per loan type
    min_income = INCOME_MIN.get(purpose, 25_000)
    if r["monthly_income"] < min_income:
        failed.append("monthly_income")
        reasons.append(
            f"Monthly income ₹{r['monthly_income']:,.0f} is below the minimum "
            f"of ₹{min_income:,} required for {r['loan_purpose']} loans."
        )

    # Rule 4: FOIR ≤ 55%
    foir = r["existing_emi"] / r["monthly_income"] if r["monthly_income"] > 0 else 1.0
    if foir > MAX_FOIR:
        failed.append("foir")
        reasons.append(
            f"FOIR {foir:.1%} exceeds the maximum allowed 55% "
            f"(existing EMI ₹{r['existing_emi']:,.0f} / "
            f"income ₹{r['monthly_income']:,.0f})."
        )

    eligible = len(failed) == 0

    return {
        "eligible": eligible,
        "applicant_name": str(r["name"]),
        "loan_purpose": str(r["loan_purpose"]),
        "reason": (
            "Applicant meets all eligibility criteria."
            if eligible
            else " | ".join(reasons)
        ),
        "failed_criteria": failed,
        "details": {
            "age": int(r["age"]),
            "credit_score": int(r["credit_score"]),
            "monthly_income": float(r["monthly_income"]),
            "existing_emi": float(r["existing_emi"]),
            "foir": round(foir, 4),
        },
    }


# ─── Tool 2: get_loan_products ────────────────────────────────────────────────
def get_loan_products(loan_purpose: str, requested_amount: float) -> dict:
    """
    Return up to 3 matching loan products for a given purpose and amount.

    Products are filtered by amount range and sorted by interest rate (lowest first).
    Each result includes a sample EMI calculated at the product's max_tenure.

    Returns:
        dict with loan_purpose, requested_amount, and list of products
    """
    purpose = loan_purpose.lower()
    products = LOAN_PRODUCTS.get(purpose, [])

    if not products:
        return {
            "error": f"No products found for loan purpose: {loan_purpose}",
            "available_purposes": list(LOAN_PRODUCTS.keys()),
        }

    # Filter by amount eligibility; fallback to all products if none match
    matching = [
        p for p in products
        if p["min_amount"] <= requested_amount <= p["max_amount"]
    ]
    if not matching:
        matching = products

    # Sort by rate ascending (best rate first)
    matching = sorted(matching, key=lambda x: x["rate"])[:3]

    results = []
    for p in matching:
        # Sample EMI using the product's max tenure
        emi_data = _calc_emi(requested_amount, p["rate"], p["max_tenure"])
        results.append({
            "product_id":            p["id"],
            "product_name":          p["name"],
            "annual_rate_pct":       p["rate"],
            "max_tenure_months":     p["max_tenure"],
            "processing_fee_pct":    p["processing_fee_pct"],
            "processing_fee_amount": round(requested_amount * p["processing_fee_pct"] / 100, 2),
            "sample_emi_at_max_tenure": emi_data["emi"],
            "total_payable_at_max_tenure": emi_data["total_payable"],
        })

    return {
        "loan_purpose":     loan_purpose,
        "requested_amount": requested_amount,
        "products":         results,
    }


# ─── Internal EMI Helper ──────────────────────────────────────────────────────
def _calc_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """
    Core EMI calculation using the reducing-balance formula.
    EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    where r = annual_rate / 12 / 100
    """
    r = annual_rate / (12 * 100)   # monthly interest rate (decimal)
    n = tenure_months

    if r == 0:
        emi = principal / n
    else:
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    total_payable  = emi * n
    total_interest = total_payable - principal

    return {
        "emi":            round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payable":  round(total_payable, 2),
    }


# ─── Tool 3: calculate_emi ────────────────────────────────────────────────────
def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> dict:
    """
    Calculate monthly EMI using the standard reducing-balance formula.
    EMI = P × r × (1+r)^n / ((1+r)^n – 1)
    where r = annual_rate / 12 / 100  (monthly rate as decimal)
          n = tenure_months

    Also returns total interest payable and total amount payable.

    Args:
        principal:      Loan amount in INR
        annual_rate:    Annual interest rate in % (e.g., 10.5 for 10.5%)
        tenure_months:  Repayment tenure in months

    Returns:
        dict with monthly_emi, total_interest_payable, total_amount_payable
    """
    result = _calc_emi(principal, annual_rate, tenure_months)

    return {
        "principal":              principal,
        "annual_rate_pct":        annual_rate,
        "tenure_months":          tenure_months,
        "monthly_emi":            result["emi"],
        "total_interest_payable": result["total_interest"],
        "total_amount_payable":   result["total_payable"],
    }


# ─── Tool 4: assess_risk_profile ─────────────────────────────────────────────
def assess_risk_profile(applicant_id: int) -> dict:
    """
    Assess an applicant's risk tier: Low / Medium / High.

    Scoring model (0–100 scale):
    ┌─────────────────────────────┬──────┬──────────────────────────────────────────┐
    │ Factor                      │ Wt.  │ Thresholds                               │
    ├─────────────────────────────┼──────┼──────────────────────────────────────────┤
    │ 1. Credit Score             │  40  │ ≥800→40, 750→35, 700→28, 650→20,        │
    │                             │      │ 600→10, <600→0                           │
    │ 2. FOIR (existing/income)   │  30  │ <20%→30, <35%→22, <50%→15,             │
    │                             │      │ <60%→8, ≥60%→0                          │
    │ 3. Employment Stability     │  20  │ ≥10yr→20, ≥5→15, ≥3→10, ≥1→6, <1→2    │
    │ 4. Loan-to-Income Ratio     │  10  │ <2x→10, <4x→7, <6x→4, ≥6x→0           │
    └─────────────────────────────┴──────┴──────────────────────────────────────────┘
    Tier: score ≥70 → Low Risk | 40–69 → Medium Risk | <40 → High Risk

    Returns:
        dict with risk_tier, risk_score, breakdown, interpretation
    """
    df = _load_data()
    row = df[df["applicant_id"] == applicant_id]

    if row.empty:
        return {"risk_tier": "Unknown", "reason": f"Applicant {applicant_id} not found."}

    r = row.iloc[0]
    score = 0
    breakdown = {}

    # Factor 1: Credit Score (40 pts)
    cs = r["credit_score"]
    if   cs >= 800: cs_pts = 40
    elif cs >= 750: cs_pts = 35
    elif cs >= 700: cs_pts = 28
    elif cs >= 650: cs_pts = 20
    elif cs >= 600: cs_pts = 10
    else:           cs_pts = 0
    score += cs_pts
    breakdown["credit_score"] = {"value": int(cs), "points": cs_pts, "max": 40}

    # Factor 2: FOIR — Debt-to-Income Ratio (30 pts)
    foir = r["existing_emi"] / r["monthly_income"] if r["monthly_income"] > 0 else 1.0
    if   foir < 0.20: foir_pts = 30
    elif foir < 0.35: foir_pts = 22
    elif foir < 0.50: foir_pts = 15
    elif foir < 0.60: foir_pts = 8
    else:             foir_pts = 0
    score += foir_pts
    breakdown["foir"] = {"value": f"{foir:.1%}", "points": foir_pts, "max": 30}

    # Factor 3: Employment Stability (20 pts)
    yrs = r["years_at_current_job"]
    if   yrs >= 10: emp_pts = 20
    elif yrs >= 5:  emp_pts = 15
    elif yrs >= 3:  emp_pts = 10
    elif yrs >= 1:  emp_pts = 6
    else:           emp_pts = 2
    score += emp_pts
    breakdown["employment_stability"] = {
        "value": f"{yrs} yrs", "points": emp_pts, "max": 20
    }

    # Factor 4: Loan-to-Income Ratio (10 pts)
    # Measures how large the requested loan is relative to annual income
    annual_income = r["monthly_income"] * 12
    lti = r["requested_amount"] / annual_income if annual_income > 0 else 999
    if   lti < 2: lti_pts = 10
    elif lti < 4: lti_pts = 7
    elif lti < 6: lti_pts = 4
    else:         lti_pts = 0
    score += lti_pts
    breakdown["loan_to_income_ratio"] = {
        "value": f"{lti:.1f}x annual income", "points": lti_pts, "max": 10
    }

    # Determine risk tier
    if   score >= 70: tier = "Low"
    elif score >= 40: tier = "Medium"
    else:             tier = "High"

    interpretations = {
        "Low":    "Strong profile — suitable for most loan products at standard rates.",
        "Medium": "Moderate risk — may require additional verification or collateral.",
        "High":   "High-risk applicant — consider rejection or mandatory collateral requirement.",
    }

    return {
        "applicant_name":  str(r["name"]),
        "risk_tier":       tier,
        "risk_score":      score,
        "breakdown":       breakdown,
        "interpretation":  f"Score {score}/100 → {tier} Risk. {interpretations[tier]}",
    }


# ─── Tool 5: get_applicant_summary ───────────────────────────────────────────
def get_applicant_summary(applicant_id: int) -> dict:
    """
    Retrieve and format all available data for an applicant from the CSV.

    Returns demographics, financial profile, loan request details,
    and derived metrics in a structured dict ready for the LLM context window.

    Returns:
        dict with complete applicant profile, or error dict if not found
    """
    df = _load_data()
    row = df[df["applicant_id"] == applicant_id]

    if row.empty:
        return {"error": f"Applicant ID {applicant_id} not found in the dataset."}

    r = row.iloc[0]

    # Derived metrics
    annual_income      = r["monthly_income"] * 12
    net_monthly        = r["monthly_income"] - r["existing_emi"]
    foir               = r["existing_emi"] / r["monthly_income"] if r["monthly_income"] > 0 else 0
    lti                = r["requested_amount"] / annual_income if annual_income > 0 else 0
    effective_amount   = r["requested_amount"] - r["down_payment"]

    return {
        # Demographics
        "applicant_id":        int(r["applicant_id"]),
        "name":                str(r["name"]),
        "age":                 int(r["age"]),
        "gender":              str(r["gender"]),
        "city":                str(r["city"]),
        # Employment
        "employment_type":     str(r["employment_type"]),
        "employer_name":       str(r["employer_name"]),
        "years_at_current_job": float(r["years_at_current_job"]),
        # Financials
        "monthly_income":      float(r["monthly_income"]),
        "annual_income":       float(annual_income),
        "existing_emi":        float(r["existing_emi"]),
        "net_monthly_surplus": float(net_monthly),
        "credit_score":        int(r["credit_score"]),
        "foir_pct":            round(foir * 100, 2),
        # Loan Request
        "loan_purpose":        str(r["loan_purpose"]),
        "requested_amount":    float(r["requested_amount"]),
        "preferred_tenure_months": int(r["preferred_tenure_months"]),
        "down_payment":        float(r["down_payment"]),
        "effective_loan_amount": float(effective_amount),
        "collateral":          str(r["collateral"]),
        "loan_to_income_ratio": round(lti, 2),
    }
