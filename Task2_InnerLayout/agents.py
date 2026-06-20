import json
import math
import llm_client
from z3 import *

from config import (
    A1_MODEL,
    A2_MODEL,
    LEFT_SETBACK,
    FRONT_SETBACK,
    RIGHT_SETBACK,
    REAR_SETBACK
)

# =============================================================================
# AGENT A1
# =============================================================================

class AgentA1:

    def __init__(self):

        self.model = A1_MODEL

    # -------------------------------------------------------------------------

    def generate_plan(self, iteration, feedback):

        prompt = """
You are an AI Architect.

Return ONLY VALID JSON. No reasoning text, no markdown, no explanation — ONLY the JSON object.

TASK:
Design an optimized house layout. Choose ONE of the following two options:

OPTION A (7-room):
- Rooms: bedroom1, bedroom2, bedroom3, bath1, bath2, kitchen, living
- Target Areas: living:350, kitchen:180, bedroom1:220, bedroom2:200, bedroom3:180, bath1:70, bath2:70
- Recommended Order: ["living", "bath1", "bedroom2", "bedroom3", "kitchen", "bedroom1", "bath2"]

OPTION B (7-room):
- Rooms: bedroom1, bedroom3, master, bath1, bath2, kitchen, living
- Target Areas: living:350, kitchen:180, bedroom1:220, bedroom3:180, master:250, bath1:70, bath2:70
- Recommended Order: ["living", "bath1", "kitchen", "bedroom1", "bedroom3", "master", "bath2"]

DIMENSION CONSTRAINTS:
- Bedrooms/master: minimum 10x10 units.
- Bathrooms: minimum 5x5 units.
- All room aspect ratios (longest/shortest side) must be < 1.8.
- SPLIT SEQUENCE: Alternate vertical/horizontal strictly. e.g. ["vertical","horizontal","vertical","horizontal","vertical","horizontal"].
- Start with "vertical" to create left/right columns. Living room goes in the left column (south/bottom side).

WALL ADJACENCY CONSTRAINTS (CRITICAL):
1. bath1 and bath2 MUST NOT share a wall.
2. kitchen and living MUST share a wall.
3. kitchen and any bathroom MUST NOT share a wall — place kitchen far from baths.
4. living MUST touch the South boundary (bottom edge).
5. Option B: bath2 must be adjacent to master.

GEOMETRY HINTS:
- South boundary (low Y) = entry side. Living room goes here.
- North boundary (high Y) = rear side.
- Tree is top-right (high X, high Y).

Iteration: {iteration}
Feedback: {feedback}

OUTPUT (return ONLY this JSON structure, nothing else):

{{"split_sequence":["vertical","horizontal","vertical","horizontal","vertical","horizontal"],"room_order":["living","bath1","kitchen","bedroom1","bedroom3","master","bath2"],"target_areas":{{"living":350,"kitchen":180,"bedroom1":220,"bedroom3":180,"master":250,"bath1":70,"bath2":70}},"adjacency":{{"living":["kitchen"],"master":["bath2"]}}}}
""".format(iteration=iteration, feedback=feedback)

        response = llm_client.chat(
            model=self.model,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )

        content = response["message"]["content"]

        print("\nRAW A1 OUTPUT:\n")
        print(content)

        # ==============================================================
        # JSON CLEANING
        # ==============================================================

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise Exception("No JSON found")

        json_text = content[start:end+1]

        try:

            proposal = json.loads(json_text)

        except Exception as e:

            print("\nJSON ERROR")
            print(e)

            print("\nBAD JSON:")
            print(json_text)

            raise Exception("Invalid JSON")

        return proposal

# =============================================================================
# AGENT A2
# =============================================================================

class AgentA2:

    def __init__(self):

        self.model = A2_MODEL

    # -------------------------------------------------------------------------

    def review(self, original_house, rooms, iteration):

        # Redefine house as the exact union of all room polygons
        # (removes empty space carved away by aspect-ratio shrinking)
        from shapely.ops import unary_union
        house = unary_union([r["polygon"] for r in rooms])
        if house.geom_type != "Polygon":
            # Fall back to convex hull if union is MultiPolygon
            house = house.convex_hull

        feedback = []

        # ==========================================================
        # Z3 LEGALITY  (check outer bounds of union house)
        # ==========================================================

        s = Solver()

        minx, miny, maxx, maxy = house.bounds

        x1 = Real("x1")
        y1 = Real("y1")
        x2 = Real("x2")
        y2 = Real("y2")

        s.add(x1 == minx)
        s.add(y1 == miny)
        s.add(x2 == maxx)
        s.add(y2 == maxy)

        s.add(x1 >= LEFT_SETBACK)
        s.add(y1 >= FRONT_SETBACK)

        s.add(x2 <= 80 - RIGHT_SETBACK)
        s.add(y2 <= 88 - REAR_SETBACK)

        legality = s.check() == sat

        # ==========================================================
        # OVERLAP CHECK
        # ==========================================================

        overlap = False

        for i in range(len(rooms)):

            for j in range(i+1, len(rooms)):

                p1 = rooms[i]["polygon"]
                p2 = rooms[j]["polygon"]

                inter = p1.intersection(p2)

                if inter.area > 1:
                    overlap = True

        # ==========================================================
        # UTILIZATION
        # ==========================================================

        room_area = sum(
            r["polygon"].area
            for r in rooms
        )

        house_area = house.area

        utilization = room_area / house_area

        dead_space = 1 - utilization

        # ==========================================================
        # COMPACTNESS
        # ==========================================================

        compactness = (
            4 * math.pi * house.area
        ) / (house.length ** 2)

        # ==========================================================
        # ADJACENCY
        # ==========================================================

        adjacency_score = 0

        living = None
        kitchen = None

        beds = []
        baths = []

        for room in rooms:

            if room["name"] == "living":
                living = room["polygon"]

            elif room["name"] == "kitchen":
                kitchen = room["polygon"]

            elif "bedroom" in room["name"]:
                beds.append(room["polygon"])

            elif "bath" in room["name"]:
                baths.append(room["polygon"])

        if living and kitchen:

            if living.touches(kitchen):
                adjacency_score += 0.4

        for bed in beds:

            for bath in baths:

                if bed.distance(bath) < 15:
                    adjacency_score += 0.1

        adjacency_score = min(adjacency_score, 1)

        # ==========================================================
        # USER CONSTRAINTS VALIDATION
        # ==========================================================

        bath1 = None
        bath2 = None
        for room in rooms:
            if room["name"] == "bath1":
                bath1 = room["polygon"]
            elif room["name"] == "bath2":
                bath2 = room["polygon"]

        bathrooms_share_wall = False
        if bath1 is not None and bath2 is not None:
            if bath1.touches(bath2):
                bathrooms_share_wall = True
                feedback.append("Bathrooms (bath1 and bath2) share a wall. They must be separated so they do not share any wall.")

        kitchen_living_share_wall = False
        if living is not None and kitchen is not None:
            if living.touches(kitchen):
                kitchen_living_share_wall = True
            else:
                feedback.append("Kitchen and living room do not share a wall. They must share a wall.")

        kitchen_bathroom_share_wall = False
        if kitchen is not None:
            if bath1 is not None and kitchen.touches(bath1):
                kitchen_bathroom_share_wall = True
                feedback.append("Kitchen and bath1 share a wall. The kitchen and bathrooms must not share any wall.")
            if bath2 is not None and kitchen.touches(bath2):
                kitchen_bathroom_share_wall = True
                feedback.append("Kitchen and bath2 share a wall. The kitchen and bathrooms must not share any wall.")

        living_on_south_boundary = False
        if living is not None:
            # Use the original buildable area min-Y as the true south edge
            orig_miny = original_house.bounds[1]
            l_minx, l_miny, l_maxx, l_maxy = living.bounds
            if abs(l_miny - orig_miny) < 0.5:
                living_on_south_boundary = True
            else:
                feedback.append("The living room is not on the South boundary of the house. The main entry is on the South side, so the living room must touch the bottom edge of the layout.")

        # Check minimum dimensions — corridor is auto-placed by geometry so skip it here
        dimension_violations = False
        aspect_ratio_violations = False
        for room in rooms:
            name = room["name"]
            poly = room["polygon"]
            rminx, rminy, rmaxx, rmaxy = poly.bounds
            width = rmaxx - rminx
            height = rmaxy - rminy

            # Corridor: no constraints (auto-filled void, any shape is valid)
            if name == "corridor":
                continue
            elif "bedroom" in name or "master" in name:
                if width < 10.0 or height < 10.0:
                    dimension_violations = True
                    feedback.append(f"Bedroom/Master dimension violation: {name} is too narrow ({width:.1f}x{height:.1f}). Minimum width and height for bedrooms/master is 10.0 units.")
            elif "bath" in name:
                if width < 5.0 or height < 5.0:
                    dimension_violations = True
                    feedback.append(f"Bathroom dimension violation: {name} is too narrow ({width:.1f}x{height:.1f}). Minimum width and height for bathrooms is 5.0 units.")

            # Aspect ratio check (corridor exempt — flexible void filler)
            min_dim = min(width, height)
            max_dim = max(width, height)
            aspect_ratio = (max_dim / min_dim) if min_dim > 0 else float('inf')
            if aspect_ratio > 1.8:
                aspect_ratio_violations = True
                feedback.append(f"Room aspect ratio violation: {name} aspect ratio is {aspect_ratio:.2f} ({width:.1f}x{height:.1f}). Aspect ratio must be less than 1.8.")

        # ==========================================================
        # SCORE
        # ==========================================================

        score = 0

        if legality:
            score += 20

        if not overlap:
            score += 20

        score += utilization * 25

        score += compactness * 15

        score += adjacency_score * 20

        # Apply user constraints penalties
        if bathrooms_share_wall:
            score -= 15
        if not kitchen_living_share_wall:
            score -= 15
        if kitchen_bathroom_share_wall:
            score -= 15
        if not living_on_south_boundary:
            score -= 15
        if dimension_violations:
            score -= 15
        if aspect_ratio_violations:
            score -= 15

        score = max(0.0, score)

        # ==========================================================
        # FEEDBACK
        # ==========================================================

        if utilization < 0.82:
            feedback.append(
                "Increase utilization."
            )

        if dead_space > 0.08:
            feedback.append(
                "Reduce dead spaces."
            )

        if adjacency_score < 0.5:
            feedback.append(
                "Improve adjacency."
            )

        if overlap:
            feedback.append(
                "Remove overlaps."
            )

        # ==========================================================
        # LLM REVIEW
        # ==========================================================

        prompt = f"""
You are an architectural reviewer.

Metrics:
Utilization: {utilization}
Dead Space: {dead_space}
Compactness: {compactness}
Adjacency: {adjacency_score}

User constraints status:
- Bathrooms share a wall: {bathrooms_share_wall} (Should be False)
- Kitchen and Living share a wall: {kitchen_living_share_wall} (Should be True)
- Kitchen and Bathroom share a wall: {kitchen_bathroom_share_wall} (Should be False)
- Living room touches South boundary: {living_on_south_boundary} (Should be True)
- Bedroom/Bathroom dimension violations: {dimension_violations} (Should be False)
- Room aspect ratio violations: {aspect_ratio_violations} (Should be False)

Give concise architectural improvement suggestions focusing on resolving any violations of the user constraints.
"""

        response = llm_client.chat(
            model=self.model,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        llm_feedback = response["message"]["content"]

        return {
            "score":score,
            "legality":legality,
            "overlap":not overlap,
            "utilization":utilization,
            "dead_space":dead_space,
            "compactness":compactness,
            "adjacency":adjacency_score,
            "feedback":feedback,
            "llm_feedback":llm_feedback
        }
