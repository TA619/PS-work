# AI Multi-Agent Architectural Planning System Documentation

This documentation serves as a comprehensive history of the design decisions, failures, successes, and technical solutions implemented throughout the development of the CAD floor plan layout generation system.

---

## 1. Development Environment & Infrastructure Evolution

### local Ollama execution (FAILED / ABANDONED)
- **Initial Setup:** Attempted to run local LLMs (Qwen-8B, Gemma-2B) on Google Colab and local hardware using Ollama.
- **Failure Cause:** Ollama requires persistent local background server processes. Colab's ephemeral runtime frequently dropped server connections. On local machines, running Qwen-8B required more RAM than available, causing frequent memory allocation crashes and slow inference times (taking up to 5 minutes per iteration).
- **Resolution:** Replaced the local Ollama dependencies with external API-Key based LLM endpoints using the OpenAI-compatible protocol (NVIDIA NIM Catalog).

### LLM Endpoint Latency & Timeout Management (SUCCESS)
- **Problems Faced:**
  - `deepseek-ai/deepseek-v4-flash` was extremely slow when reasoning was enabled (`"thinking": True`), triggering 120-second read timeouts.
  - Large models like `qwen/qwen3.5-122b-a10b` frequently hung under heavy API load, leading to network timeouts.
- **Resolution:**
  - Standardized Agent A1 on **`mistralai/mistral-small-4-119b-2603`** (Mistral Small) and Agent A2 on **`meta/llama-3.1-8b-instruct`** (Llama 3.1 8B), which are fast, reliable, and highly conformant to structured JSON output.
  - Created a custom, zero-dependency `llm_client.py` using Python's standard `urllib.request` supporting timeouts (120s) and automatic retries (up to 3 times) with exponential backoff on HTTP errors.

### Windows Encoding Compatibility (SUCCESS)
- **Problems Faced:** Windows terminals crashed with `UnicodeEncodeError` when trying to print LLM response characters (e.g. non-breaking hyphens `‑` or quote marks) in standard streams.
- **Resolution:** Explicitly reconfigured standard output and error streams (`sys.stdout/stderr.reconfigure(encoding='utf-8')`) at startup in `main.py`.

---

## 2. Layout Generation & Partitioning Evolution

### Phase 1: Fully Autonomous Python Code Generation (FAILED)
- **Concept:** Agent A1 dynamically generated raw Python code using `shapely` and `ezdxf` to place rooms and export CAD drawings.
- **Failure Cause:** The LLM frequently produced malformed geometry code, overlapping room coordinates, invalid operations (like subtracting intersecting polygons causing empty results), and infinite clipping loops. Debugging dynamic runtime errors from LLM-generated code was impossible.
- **Resolution:** Abandoned code generation and introduced a deterministic CAD engine that reads structured parameters.

### Phase 2: Row-Based Split Ratios (FAILED)
- **Concept:** Agent A1 output target height/width ratios, and the engine parsed them to split the layout.
- **Failure Cause:** The model regularly omitted required parameters (resulting in `KeyError` crashes like missing `height_ratio`), and room placements were clustered randomly, leaving huge dead spaces and non-standard room shapes.

### Phase 3: Standard Binary Space Partitioning (BSP) (PARTIAL SUCCESS)
- **Concept:** Procedural CAD engine dividing spaces based on Agent A1's room ordering and target areas using recursive halving (`half = len(rooms) // 2`).
- **Success:** Solved the random clustering problem. Interior layouts became compact and guaranteed no room overlaps.
- **Failure Cause:** Guest bathroom (`bath1`) became a highly stretched, narrow vertical strip (e.g., 5x30 units), which was architecturally unrealistic and unusable.

### Phase 4: Custom BSP Splits & Hallway Integration (Option A - SUCCESS)
- **Concept:** Added a `hallway` room directly above `bath1` to connect the living room and bedroom.
- **Solution:** Modified the `recursive_split` algorithm in `cad_engine.py` to allow custom, non-midpoint splits:
  - If `"living"` is the first room in a sub-list, it splits off index 1 (leaving the rest of the column intact).
  - If `"bedroom2"` is the last room, it splits off the last index.
  - Stacks `hallway` and `bath1` horizontally in the middle column so that `bath1` takes 70% of the height (making it square-ish) and the hallway takes 30% (as a horizontal passage).

### Phase 5: Multi-Layout Blueprint Support (Option B - SUCCESS)
- **Concept:** Integrated a new 7-room sketch layout suggestion containing 1 kitchen, 2 living room, 3/4 bedrooms, 5 master bedroom, 6 guest bathroom (common), and 7 attached bathroom (attached).
- **Problems Faced:** Stacking the master suite vertically would make the attached bath a vertical strip. Hardcoding Option B would delete the Option A hallway layout.
- **Solution:**
  - Modified the CAD engine split logic to dynamically detect whether `"hallway"` is in the rooms list. If present, it applies the Option A splits; otherwise, it applies Option B splits.
  - Added a direction override so that when splitting `["master", "bath2"]`, it overrides the sequence to split horizontally, stacking the attached bath on top of the master bedroom.
  - Updated Agent A1's prompt to offer both Option A and Option B as valid active blueprint choices.

---

## 3. Doors & Windows Rendering Evolution

### Phase 1: Doors on All Intersections (FAILED)
- **Concept:** The CAD exporter looped through all adjacent room pairs and drew a door on every shared boundary segment.
- **Failure Cause:** Too many doors were generated (e.g. doors between adjacent bedrooms, doors directly into kitchens from bedrooms), which is highly unrealistic and compromises privacy.

### Phase 2: Priority-Based Single-Door Connections (SUCCESS)
- **Concept:** Each room (except `living`) must receive exactly one entrance door on the wall sharing with the highest priority adjacent space.
- **Solution:**
  - Implemented a priority order list: `Living Room > Hallway > Kitchen > Bedrooms > Bathrooms`.
  - For each room (except `living`), the CAD engine evaluates all adjacent room boundaries, selects the touching room with the highest priority rank, and draws a single door on that segment.
  - Added a `drawn_doors` cache to prevent drawing duplicate door lines on the same wall.
  - Changed doors to be drawn closed and collinear (overlapping) with the walls in **Red** (color 1) for a cleaner layout.

### Phase 3: External Boundary Windows (SUCCESS)
- **Concept:** Draw a window on the exterior wall of each room that touches the house boundary.
- **Solution:**
  - Computes the intersection of each room's boundary with the house exterior boundary.
  - Identifies the longest straight segment of the intersection and centers a **Dark Blue** (color 5) window line of length 4.
  - For the living room, it skips the South boundary segment to prevent the window from overlapping the main entrance door.

---

## 4. Optimization & Scoring Convergence (SUCCESS)

- **Problems Faced:** With more rooms and custom split structures, the verifier's adjacency score (based on the distance between all bedrooms and bathrooms) was mathematically capped below 1.0. This prevented the orchestrator from reaching its hardcoded target score of 91, resulting in infinite loops and API timeouts.
- **Solution:** Lowered the `TARGET_SCORE` to 85 in `config.py`. The generation loop now converges immediately on the first iteration once all legal, dimension, and adjacency constraints are satisfied, yielding successful CAD drawings in seconds.
