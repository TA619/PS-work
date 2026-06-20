import sys

# Force stdout/stderr to use UTF-8 to prevent UnicodeEncodeErrors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from orchestrator import Orchestrator
from config import LLM_PROVIDER, A1_MODEL, A2_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("="*80)
    print("AI MULTI AGENT ARCHITECTURAL PLANNING SYSTEM")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"- Provider: {LLM_PROVIDER.upper()}")
    print(f"- Agent A1 Model: {A1_MODEL}")
    print(f"- Agent A2 Model: {A2_MODEL}")

    has_key = False
    if LLM_PROVIDER == "openai":
        has_key = bool(OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        has_key = bool(ANTHROPIC_API_KEY)
    elif LLM_PROVIDER == "gemini":
        has_key = bool(GEMINI_API_KEY)

    if not has_key:
        print(f"\n[ERROR] API Key for provider '{LLM_PROVIDER.upper()}' is missing!")
        print("Please check your .env file and set the appropriate API key.")
        sys.exit(1)

    print(f"[OK] LLM Configuration Validated (API Key is set)")

    Orchestrator().run()

