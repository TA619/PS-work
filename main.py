import ollama

from orchestrator import Orchestrator

# ============================================================================
# VERIFY OLLAMA
# ============================================================================

print("=" * 80)
print("VERIFYING OLLAMA")
print("=" * 80)

try:

    ollama.list()

    print("✅ OLLAMA CONNECTED")

except Exception as e:

    print("❌ OLLAMA ERROR")
    print(e)

    raise SystemExit()

# ============================================================================
# RUN
# ============================================================================

print("\n" + "=" * 80)
print("STARTING LOW RAM TWO AGENT SYSTEM")
print("=" * 80)

orchestrator = Orchestrator()

orchestrator.run()

print("\n✅ FINISHED")
