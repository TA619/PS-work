import ezdxf

from shapely.geometry import (
    Polygon,
    Point,
    box
)

from shapely.ops import unary_union

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
# CAD ENGINE
# ============================================================================

class CADEngine:

    def generate_house(self,
                       proposal,
                       iteration):

        split_x = proposal.split_x
        right_height = proposal.right_height

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

        tree_circle = Point(
            TREE_CENTER
        ).buffer(
            TREE_RADIUS
        )

        # =====================================================
        # DYNAMIC L SHAPE
        # =====================================================

        left_rect = box(
            LEFT_SETBACK,
            FRONT_SETBACK,
            split_x,
            MAX_Y - REAR_SETBACK
        )

        right_rect = box(
            split_x,
            FRONT_SETBACK,
            MAX_X - RIGHT_SETBACK,
            right_height
        )

        house = unary_union([
            left_rect,
            right_rect
        ])

        # =====================================================
        # VALIDATION
        # =====================================================

        if house.intersects(tree_circle):

            return None

        if not buildable.contains(house):

            return None

        if not house.is_valid:

            return None

        # =====================================================
        # CREATE DXF
        # =====================================================

        filename = (
            f"house_iteration_{iteration}.dxf"
        )

        doc = ezdxf.new()

        msp = doc.modelspace()

        # =====================================================
        # PLOT BOUNDARY
        # =====================================================

        plot_closed = (
            PLOT_COORDS +
            [PLOT_COORDS[0]]
        )

        msp.add_lwpolyline(
            plot_closed,
            close=True,
            dxfattribs={
                "color": 1
            }
        )

        # =====================================================
        # TREE CIRCLE
        # =====================================================

        msp.add_circle(
            center=TREE_CENTER,
            radius=TREE_RADIUS,
            dxfattribs={
                "color": 3
            }
        )

        # =====================================================
        # HOUSE OUTLINE
        # =====================================================

        coords = list(
            house.exterior.coords
        )

        msp.add_lwpolyline(
            coords,
            close=True,
            dxfattribs={
                "color": 2
            }
        )

        # =====================================================
        # ENTRY DOOR
        # =====================================================

        door_x = LEFT_SETBACK + 5

        msp.add_line(
            (door_x, FRONT_SETBACK),
            (door_x + 3, FRONT_SETBACK),
            dxfattribs={"color": 5}
        )

        # =====================================================
        # LABEL
        # =====================================================

        msp.add_text(
            f"AREA = {round(house.area,2)}",
            dxfattribs={
                "height": 2
            }
        ).set_placement(
            (10, 10)
        )

        # =====================================================
        # SAVE
        # =====================================================

        doc.saveas(filename)

        return {
            "house": house,
            "filename": filename
        }
