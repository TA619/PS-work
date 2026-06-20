# AI Multi-Agent Architectural Planning System

An automated floor plan layout generation and optimization system. It combines recursive Binary Space Partitioning (BSP), a Z3 mathematical solver, and two Large Language Model agents (Agent A1 Architect and Agent A2 Reviewer) to design, evaluate, and iteratively optimize house floor plans.

---

## Features

### 1. Two Active Blueprint Suggestions
- **Option A (8-room Hallway Layout):**
  - Stacks a horizontal `hallway` directly above `bath1` to connect the `living` room to `bedroom2`, resolving narrow bathroom shapes.
  - Recommended order: `["living", "bath1", "hallway", "bedroom2", "bedroom3", "kitchen", "bedroom1", "bath2"]`
- **Option B (7-room Sketch Layout):**
  - Mimics a corner-suite layout with a `master` bedroom on the East side.
  - Stacks `bedroom3` on top of `bedroom1` horizontally in the middle.
  - Stacks attached `bath2` on top of `master` horizontally on the right side using a direction override.
  - Recommended order: `["living", "bath1", "kitchen", "bedroom1", "bedroom3", "master", "bath2"]`

### 2. Intelligent Door & Window Rendering
- **Priority-Based Closed Doors:** Doors are drawn closed and collinear (overlapping) with the walls in **Red** (color 1). Each room (except living room) is assigned exactly one entrance door on the wall sharing with the highest priority adjacent room (`Living > Hallway > Kitchen > Bedrooms > Bathrooms`).
- **Dark Blue Windows:** A single window (length 4) is centered on the longest exterior boundary segment of each room in **Dark Blue** (color 5). The living room window is placed on the West boundary to avoid overlapping the South entrance door.

### 3. Unified Zero-Dependency LLM Client
- Built-in lightweight client in `llm_client.py` using Python's standard `urllib.request`. Supports OpenAI, NVIDIA NIM Catalog, Anthropic, and Gemini out of the box with automatic exponential backoff retries.

---

## Color Codes
- **Red (color 1):** Plot coordinates boundary & doors (thresholds and closed panels)
- **Yellow (color 2):** House setback boundaries
- **Green (color 3):** Room boundaries & room name labels
- **Dark Blue (color 5):** Windows & Tree circle boundary

---

## Prerequisites

1. **Python 3.x** installed on your system.
2. Install the required Python libraries using `pip`:
   ```bash
   pip install ezdxf shapely z3-solver
   ```
3. Get an API Key for one of the supported LLM providers (default is NVIDIA NIM API Catalog, which uses the OpenAI-compatible base).

---

## Installation & Setup

1. Copy the `.env.template` file to `.env`:
   ```bash
   copy .env.template .env
   ```
2. Open the `.env` file and configure your LLM provider and API key:
   - To use NVIDIA Catalog, set:
     ```env
     LLM_PROVIDER=openai
     A1_MODEL=mistralai/mistral-small-4-119b-2603
     A2_MODEL=meta/llama-3.1-8b-instruct
     OPENAI_API_KEY=your_nvapi_key_here
     OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
     ```

---

## How to Run

Run the orchestrator:
```bash
python main.py
```

### Output
During execution, the orchestrator will output the multi-agent reasoning logs and save the final CAD floor plan in DXF format:
- `house_iteration_1.dxf`
- `house_iteration_2.dxf`
- *etc.*

Open the `.dxf` files using any CAD software (such as AutoCAD, LibreCAD) or an online DXF viewer.
