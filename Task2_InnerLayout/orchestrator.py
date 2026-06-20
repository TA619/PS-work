import json

from config import MAX_ITERATIONS, TARGET_SCORE
from agents import AgentA1, AgentA2
from cad_engine import CADEngine

# =============================================================================
# ORCHESTRATOR
# =============================================================================

class Orchestrator:

    def __init__(self):

        self.a1 = AgentA1()

        self.a2 = AgentA2()

        self.engine = CADEngine()

    # -------------------------------------------------------------------------

    def run(self):

        feedback = "No previous feedback."

        for iteration in range(1, MAX_ITERATIONS+1):

            print("\n" + "#"*80)
            print(f"ITERATION {iteration}")
            print("#"*80)

            # ======================================================
            # A1
            # ======================================================

            proposal = self.a1.generate_plan(
                iteration,
                feedback
            )

            print("\nA1 PROPOSAL:")
            print(json.dumps(proposal, indent=2))

            # ======================================================
            # HOUSE
            # ======================================================

            house = self.engine.create_house()

            # ======================================================
            # ROOMS
            # ======================================================

            rooms = self.engine.create_rooms(
                house,
                proposal
            )

            # ======================================================
            # EXPORT
            # ======================================================

            dxf_file = self.engine.export_dxf(
                house,
                rooms,
                iteration
            )

            # ======================================================
            # A2
            # ======================================================

            result = self.a2.review(
                house,
                rooms,
                iteration
            )

            # ======================================================
            # RESULTS
            # ======================================================

            print("\n" + "="*70)
            print("VERIFICATION RESULTS")
            print("="*70)

            print(f"\nScore: {result['score']:.2f}")
            print(f"Legality: {result['legality']}")
            print(f"No Overlap: {result['overlap']}")
            print(f"Utilization: {result['utilization']:.2f}")
            print(f"Dead Space: {result['dead_space']:.2f}")
            print(f"Compactness: {result['compactness']:.2f}")
            print(f"Adjacency: {result['adjacency']:.2f}")

            print("\nFeedback:")

            for f in result["feedback"]:
                print("-", f)

            if "llm_feedback" in result:
                print("\nSuggestions:")
                print(result["llm_feedback"])

            print(f"\nDXF FILE: {dxf_file}")

            # ======================================================
            # SUCCESS
            # ======================================================

            if result["score"] >= TARGET_SCORE:

                print("\n[SUCCESS]")
                print(f"Converged in {iteration} iterations")

                break

            feedback = "\n".join(
                result["feedback"]
            )
