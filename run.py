"""
CreditSage AI Advisor — Entry Point

Run this file to launch the Streamlit application:
    python run.py

Or directly:
    streamlit run app.py
"""

import subprocess
import sys
import os


def main():
    print("=" * 55)
    print("  💳  CreditSage AI Advisor — Loan Advisory Agent")
    print("=" * 55)

    # ── Pre-flight checks ──────────────────────────────────────
    errors = []

    # Check for .env / API key
    if not os.path.exists(".env") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: No .env file found and OPENAI_API_KEY is not set.")
        print("   Create a .env file based on .env.example before running.")
    else:
        print("✅  Environment variables detected.")

    # Check for dataset
    if not os.path.exists("creditsage_loan_applications.csv"):
        errors.append(
            "❌  Dataset missing: creditsage_loan_applications.csv not found.\n"
            "   Place the CSV file in the project root directory."
        )
    else:
        print("✅  Dataset file found.")

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)

    print("\n🚀  Launching Streamlit app…")
    print("   Local URL: http://localhost:8501\n")

    # ── Launch Streamlit ───────────────────────────────────────
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.port", "8501",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
