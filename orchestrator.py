from config import MAX_ITERATIONS
from agent_a1 import AgentA1
from cad_engine import CADEngine
from agent_a2 import AgentA2

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class Orchestrator:

    def __init__(self):

        self.a1 = AgentA1()

        self.engine = CADEngine()

        self.a2 = AgentA2()

    def run(self):

        feedback = None

        for iteration in range(
            1,
            MAX_ITERATIONS + 1
        ):

            print("\n" + "#" * 80)

            print(
                f"ITERATION {iteration}"
            )

            print("#" * 80)

            try:

                # =================================================
                # A1
                # =================================================

                proposal = (
                    self.a1.generate_layout(
                        feedback
                    )
                )

                print(
                    f"\nA1 Proposal:"
                    f"\nsplit_x = {proposal.split_x}"
                    f"\nright_height = {proposal.right_height}"
                )

                # =================================================
                # CAD ENGINE
                # =================================================

                result = (
                    self.engine.generate_house(
                        proposal,
                        iteration
                    )
                )

                if result is None:

                    print(
                        "\n❌ Invalid geometry"
                    )

                    feedback = (
                        "Geometry invalid."
                    )

                    continue

                house = result["house"]

                print(
                    f"\n✅ DXF Generated:"
                    f"\n{result['filename']}"
                )

                # =================================================
                # A2
                # =================================================

                verification = (
                    self.a2.verify(
                        house
                    )
                )

                # =================================================
                # RESULTS
                # =================================================

                print("\n" + "=" * 70)

                print("VERIFICATION RESULTS")

                print("=" * 70)

                print(
                    "\nTree Preservation:",
                    verification["tree_ok"]
                )

                print(
                    "Setbacks:",
                    verification["setbacks_ok"]
                )

                print(
                    "Geometry:",
                    verification["geometry_ok"]
                )

                print(
                    "Area Optimization:",
                    verification["area_ok"]
                )

                print(
                    "Z3 Constraints:",
                    verification["z3_ok"]
                )

                print(
                    "\nTOTAL:",
                    verification["total"]
                )

                print(
                    "STATUS:",
                    verification["status"]
                )

                print(
                    "HOUSE AREA:",
                    round(
                        verification["area"],
                        2
                    )
                )

                # =================================================
                # SUCCESS
                # =================================================

                if verification["status"] == "PASS":

                    print("\n🎉 SUCCESS")

                    print(
                        f"\nFinal DXF:"
                        f"\nhouse_iteration_{iteration}.dxf"
                    )

                    break

                # =================================================
                # FEEDBACK
                # =================================================

                feedback = (
                    f"Area too low:"
                    f" {verification['area']}"
                )

            except Exception as e:

                print("\n❌ ERROR")

                print(e)
