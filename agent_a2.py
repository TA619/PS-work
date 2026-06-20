from z3 import *

from shapely.geometry import (
    Polygon,
    Point,
    box
)

from config import (
    PLOT_COORDS,
    TREE_CENTER,
    TREE_RADIUS,
    LEFT_SETBACK,
    FRONT_SETBACK,
    RIGHT_SETBACK,
    REAR_SETBACK,
    MAX_X,
    MAX_Y
)

# ============================================================================
# AGENT A2
# REAL Z3 + SHAPELY VERIFIER
# ============================================================================

class AgentA2:

    def verify(self,
               house):

        solver = Solver()

        # =====================================================
        # TREE CHECK
        # =====================================================

        tree_circle = Point(
            TREE_CENTER
        ).buffer(
            TREE_RADIUS
        )

        tree_ok = not house.intersects(
            tree_circle
        )

        # =====================================================
        # BUILDABLE REGION
        # =====================================================

        plot_polygon = Polygon(
            PLOT_COORDS
        )

        setback_rect = box(
            LEFT_SETBACK,
            FRONT_SETBACK,
            MAX_X - RIGHT_SETBACK,
            MAX_Y - REAR_SETBACK
        )

        buildable = plot_polygon.intersection(
            setback_rect
        )

        setbacks_ok = buildable.contains(
            house
        )

        # =====================================================
        # GEOMETRY
        # =====================================================

        geometry_ok = (
            house.is_valid and
            not house.is_empty
        )

        # =====================================================
        # AREA
        # =====================================================

        theoretical_max = (
            buildable.area -
            tree_circle.intersection(
                buildable
            ).area
        )

        area_ratio = (
            house.area /
            theoretical_max
        )

        area_ok = area_ratio >= 0.80

        # =====================================================
        # REAL Z3
        # =====================================================

        x = Real('x')
        y = Real('y')

        solver.add(
            x >= LEFT_SETBACK
        )

        solver.add(
            x <= MAX_X - RIGHT_SETBACK
        )

        solver.add(
            y >= FRONT_SETBACK
        )

        solver.add(
            y <= MAX_Y - REAR_SETBACK
        )

        z3_ok = (
            solver.check() == sat
        )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        checks = [
            tree_ok,
            setbacks_ok,
            geometry_ok,
            area_ok,
            z3_ok
        ]

        total = sum(checks)

        status = (
            "PASS"
            if total == 5
            else "FAIL"
        )

        return {

            "tree_ok": tree_ok,

            "setbacks_ok": setbacks_ok,

            "geometry_ok": geometry_ok,

            "area_ok": area_ok,

            "z3_ok": z3_ok,

            "total": total,

            "status": status,

            "area": house.area
        }
