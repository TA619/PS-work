import os

# =============================================================================
# ENVIRONMENT LOADER
# =============================================================================

def load_env(env_path=".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

# Load .env variables into environment
load_env()

# =============================================================================
# CONFIG
# =============================================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
A1_MODEL = os.getenv("A1_MODEL", "gpt-4o-mini")
A2_MODEL = os.getenv("A2_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

MAX_ITERATIONS = 8
TARGET_SCORE = 85


PLOT_COORDS = [
    (0,8),
    (35,8),
    (35,0),
    (75,0),
    (75,8),
    (80,8),
    (80,88),
    (75,88),
    (75,84),
    (0,84)
]

TREE_CENTER = (75,88)
TREE_RADIUS = 12

FRONT_SETBACK = 20
REAR_SETBACK = 15
LEFT_SETBACK = 5
RIGHT_SETBACK = 5
