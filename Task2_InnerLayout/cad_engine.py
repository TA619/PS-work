import ezdxf
from shapely.geometry import Polygon, Point, box

from config import (
    PLOT_COORDS,
    TREE_CENTER,
    TREE_RADIUS,
    LEFT_SETBACK,
    FRONT_SETBACK,
    RIGHT_SETBACK,
    REAR_SETBACK
)
from bsp_node import BSPNode

# =============================================================================
# CAD ENGINE
# =============================================================================

class CADEngine:

    def __init__(self):

        self.plot_polygon = Polygon(PLOT_COORDS)

        setback_box = box(
            LEFT_SETBACK,
            FRONT_SETBACK,
            80 - RIGHT_SETBACK,
            88 - REAR_SETBACK
        )

        buildable = self.plot_polygon.intersection(setback_box)

        tree_circle = Point(TREE_CENTER).buffer(TREE_RADIUS)

        self.legal_area = buildable.difference(tree_circle)

    # -------------------------------------------------------------------------

    def create_house(self):

        return self.legal_area

    # -------------------------------------------------------------------------

    def recursive_split(
        self,
        node,
        rooms,
        target_areas,
        split_sequence,
        depth=0
    ):

        if len(rooms) == 1:

            node.room = rooms[0]
            return

        rect = node.rect

        minx, miny, maxx, maxy = rect.bounds

        width = maxx - minx
        height = maxy - miny

        split_type = split_sequence[
            depth % len(split_sequence)
        ]

        # Override split type for master/bath2 to split horizontally
        if "master" in rooms and "bath2" in rooms and len(rooms) == 2:
            split_type = "horizontal"

        # Standard uniform splitting
        half = len(rooms)//2

        left_rooms = rooms[:half]
        right_rooms = rooms[half:]

        left_area = sum(
            target_areas[r]
            for r in left_rooms
        )

        right_area = sum(
            target_areas[r]
            for r in right_rooms
        )

        total = left_area + right_area

        ratio = left_area / total

        if split_type == "vertical":

            split_x = minx + width * ratio

            rect1 = box(
                minx,
                miny,
                split_x,
                maxy
            )

            rect2 = box(
                split_x,
                miny,
                maxx,
                maxy
            )

        else:

            split_y = miny + height * ratio

            rect1 = box(
                minx,
                miny,
                maxx,
                split_y
            )

            rect2 = box(
                minx,
                split_y,
                maxx,
                maxy
            )

        node.left = BSPNode(rect1)
        node.right = BSPNode(rect2)

        self.recursive_split(
            node.left,
            left_rooms,
            target_areas,
            split_sequence,
            depth + 1
        )

        self.recursive_split(
            node.right,
            right_rooms,
            target_areas,
            split_sequence,
            depth + 1
        )

    # -------------------------------------------------------------------------

    def collect_rooms(self, node, house, rooms):

        if node is None:
            return

        if node.room is not None:

            poly = node.rect.intersection(house)

            if (
                not poly.is_empty and
                poly.area > 40 and
                poly.geom_type == "Polygon"
            ):

                rooms.append({
                    "name":node.room,
                    "polygon":poly
                })

        self.collect_rooms(
            node.left,
            house,
            rooms
        )

        self.collect_rooms(
            node.right,
            house,
            rooms
        )

    # -------------------------------------------------------------------------

    def create_rooms(self, house, proposal):

        room_order    = list(proposal["room_order"])
        target_areas  = dict(proposal["target_areas"])
        split_sequence = proposal["split_sequence"]

        # ------------------------------------------------------------------
        # BSP SPLIT — use the LLM-provided room order directly
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # PRE-PROCESS: ensure no bath sits immediately next to kitchen
        # in the BSP room_order. If one does, swap it with the nearest
        # bedroom/master that is not already a kitchen-neighbor.
        # This changes the layout topology, not the geometry.
        # ------------------------------------------------------------------
        kitchen_idx = next(
            (i for i, r in enumerate(room_order) if r == "kitchen"), None
        )
        if kitchen_idx is not None:
            for offset in [-1, 1]:
                ni = kitchen_idx + offset
                if 0 <= ni < len(room_order) and "bath" in room_order[ni]:
                    # Find a bedroom/master far from kitchen to swap with
                    swap_idx = next(
                        (
                            i for i, r in enumerate(room_order)
                            if ("bedroom" in r or r == "master")
                            and abs(i - kitchen_idx) > 1
                        ),
                        None,
                    )
                    if swap_idx is not None:
                        bath_name = room_order[ni]
                        bed_name  = room_order[swap_idx]
                        room_order[ni], room_order[swap_idx] = (
                            room_order[swap_idx],
                            room_order[ni],
                        )
                        print(
                            f"[FIX] Separated bath from kitchen: "
                            f"swapped {bath_name} ↔ {bed_name} in room_order"
                        )

        root = BSPNode(house)

        self.recursive_split(
            root,
            room_order,
            target_areas,
            split_sequence
        )

        rooms = []

        self.collect_rooms(root, house, rooms)


        # ------------------------------------------------------------------
        # POST-PROCESS 1: shrink rooms whose aspect ratio exceeds 1.6
        # Corridor (if already in rooms) is exempt — any shape is fine.
        # ------------------------------------------------------------------
        hminx, hminy, hmaxx, hmaxy = house.bounds
        for r in rooms:
            if r["name"] == "corridor":
                continue
            poly = r["polygon"]
            if poly.geom_type != "Polygon":
                continue
            rminx, rminy, rmaxx, rmaxy = poly.bounds
            width  = rmaxx - rminx
            height = rmaxy - rminy

            if min(width, height) <= 0:
                continue

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 1.6:
                target_ratio = 1.5

                if width > height * 1.6:
                    new_width     = height * target_ratio
                    shrink_amount = width - new_width

                    is_left_boundary  = abs(rminx - hminx) < 0.2
                    is_right_boundary = abs(rmaxx - hmaxx) < 0.2

                    if is_left_boundary and is_right_boundary:
                        rminx += shrink_amount / 2
                        rmaxx -= shrink_amount / 2
                    elif is_left_boundary:
                        rminx += shrink_amount
                    elif is_right_boundary:
                        rmaxx -= shrink_amount
                    else:
                        rminx += shrink_amount / 2
                        rmaxx -= shrink_amount / 2

                elif height > width * 1.6:
                    new_height    = width * target_ratio
                    shrink_amount = height - new_height

                    is_bottom_boundary = abs(rminy - hminy) < 0.2
                    is_top_boundary    = abs(rmaxy - hmaxy) < 0.2

                    if is_bottom_boundary and is_top_boundary:
                        rmaxy -= shrink_amount
                    elif is_top_boundary:
                        rmaxy -= shrink_amount
                    elif is_bottom_boundary:
                        rminy += shrink_amount
                    else:
                        rminy += shrink_amount / 2
                        rmaxy -= shrink_amount / 2

                from shapely.geometry import box as _box
                shrunk_box = _box(rminx, rminy, rmaxx, rmaxy)
                new_poly = poly.intersection(shrunk_box)
                if not new_poly.is_empty and new_poly.geom_type == "Polygon":
                    r["polygon"] = new_poly

        # ------------------------------------------------------------------
        # POST-PROCESS 2: Geometric void handling (priority-ordered rules)
        #
        # After shrinking, void = buildable area minus all rooms.
        # Each void polygon is handled by the first matching rule:
        #
        #   Rule A — void adjacent to master bedroom
        #            → merge into master (expand master, no new room)
        #
        #   Rule B — void adjacent to bath1 (guest washroom)
        #            → always label as "corridor" so guest bath always
        #              has a dedicated circulation/transition space
        #
        #   Rule C — void adjacent to 2+ named rooms (general dead space)
        #            → label as "corridor"
        #
        # Voids are processed largest-first so the biggest gap is handled first.
        # ------------------------------------------------------------------
        from shapely.ops import unary_union as _uu

        room_union = _uu([r["polygon"] for r in rooms])
        void       = house.difference(room_union)

        void_polys = []
        if not void.is_empty:
            if void.geom_type == "Polygon":
                void_polys = [void]
            elif void.geom_type == "MultiPolygon":
                void_polys = list(void.geoms)

        # Sort largest void first
        void_polys.sort(key=lambda p: p.area, reverse=True)

        for vp in void_polys:
            if vp.area < 40:
                continue

            # Find which rooms share a meaningful wall with this void
            adjacent_names = []
            for r in rooms:
                try:
                    shared = vp.intersection(r["polygon"])
                    if shared.length > 2.0:
                        adjacent_names.append(r["name"])
                except Exception:
                    pass

            if not adjacent_names:
                continue

            # ── Rule A: void touches master → expand master bedroom ──────────
            if "master" in adjacent_names:
                for r in rooms:
                    if r["name"] == "master":
                        merged = r["polygon"].union(vp)
                        if merged.geom_type == "Polygon":
                            r["polygon"] = merged
                            print(
                                f"[AUTO] Merged void (area={vp.area:.1f}) "
                                f"into master bedroom"
                            )
                        break
                continue  # void consumed — move to next

            # ── Rule B: void touches bath1 (guest washroom) → corridor ───────
            if "bath1" in adjacent_names:
                rooms.append({"name": "corridor", "polygon": vp})
                print(
                    f"[AUTO] Inserted guest corridor near bath1 "
                    f"(area={vp.area:.1f}, adjacent to: {adjacent_names})"
                )
                continue  # void consumed

            # ── Rule C: void touches 2+ rooms → general corridor ─────────────
            if len(adjacent_names) >= 2:
                if not any(r["name"] == "corridor" for r in rooms):
                    rooms.append({"name": "corridor", "polygon": vp})
                    print(
                        f"[AUTO] Inserted corridor in void "
                        f"(area={vp.area:.1f}, adjacent to: {adjacent_names})"
                    )
        # ── Post-void: extend living room downward into any remaining void ────
        # The void directly below living (same x-column, lower y) is only
        # adjacent to living itself so Rules A/B/C won't consume it.
        # Extend living southward to fill it, making the house more rectangular.
        living_r2 = next((r for r in rooms if r["name"] == "living"), None)
        if living_r2:
            from shapely.geometry import box as _bx
            from shapely.ops import unary_union as _uu2
            l_poly2 = living_r2["polygon"]
            lminx2, lminy2, lmaxx2, lmaxy2 = l_poly2.bounds
            # Void remaining after all rules
            remaining_union = _uu2([r["polygon"] for r in rooms])
            remaining_void  = house.difference(remaining_union)
            if not remaining_void.is_empty:
                # Clip to the column directly below living
                col_below = _bx(lminx2, house.bounds[1], lmaxx2, lminy2)
                void_below = remaining_void.intersection(col_below)
                if not void_below.is_empty and void_below.area > 5:
                    new_living = l_poly2.union(void_below)
                    if new_living.geom_type == "Polygon":
                        living_r2["polygon"] = new_living
                        print(
                            f"[EXTEND] Expanded living room downward "
                            f"(+{void_below.area:.1f} sq units)"
                        )

        return rooms



    # -------------------------------------------------------------------------

    def export_dxf(self, house, rooms, iteration):

        from shapely.ops import unary_union
        house = unary_union([r["polygon"] for r in rooms])

        # Pre-compute living room polygon for south-wall window skip
        living = next((r["polygon"] for r in rooms if r["name"] == "living"), None)

        doc = ezdxf.new()

        msp = doc.modelspace()

        # ==========================================================
        # PLOT
        # ==========================================================

        plot_closed = PLOT_COORDS + [PLOT_COORDS[0]]

        msp.add_lwpolyline(
            plot_closed,
            dxfattribs={"color":1}
        )

        # ==========================================================
        # HOUSE
        # ==========================================================

        if house.geom_type == "Polygon":

            msp.add_lwpolyline(
                list(house.exterior.coords),
                dxfattribs={"color":2}
            )

        # ==========================================================
        # ROOMS
        # ==========================================================

        for room in rooms:

            poly = room["polygon"]

            if poly.geom_type != "Polygon":
                continue

            coords = list(poly.exterior.coords)

            msp.add_lwpolyline(
                coords,
                dxfattribs={"color":3}
            )

            center = poly.centroid

            # Draw the room name slightly higher
            msp.add_text(
                room["name"].upper(),
                height=2
            ).set_placement(
                (center.x, center.y + 1.2)
            )

            # Calculate and draw the room dimensions slightly lower
            rminx, rminy, rmaxx, rmaxy = poly.bounds
            width = rmaxx - rminx
            height = rmaxy - rminy
            dim_text = f"{width:.1f}x{height:.1f}"
            
            msp.add_text(
                dim_text,
                height=1.5
            ).set_placement(
                (center.x, center.y - 1.2)
            )

            # Draw a window on exterior boundary walls — skip corridor and hallway (interior circulation)
            if room["name"] not in ("hallway", "corridor"):
                house_b = house.boundary
                room_b = poly.boundary
                inter = room_b.intersection(house_b)
                
                if not inter.is_empty:
                    segments = []
                    if inter.geom_type == "LineString":
                        segments.append(inter)
                    elif inter.geom_type == "MultiLineString":
                        for line in inter.geoms:
                            segments.append(line)
                    elif inter.geom_type == "GeometryCollection":
                        for geom in inter.geoms:
                            if geom.geom_type == "LineString":
                                segments.append(geom)
                            elif geom.geom_type == "MultiLineString":
                                for line in geom.geoms:
                                    segments.append(line)
                    
                    longest_seg = None
                    max_len = 0
                    for seg in segments:
                        if seg.length > max_len:
                            max_len = seg.length
                            longest_seg = seg
                            
                    if longest_seg is not None and max_len > 4.0:
                        coords_seg = list(longest_seg.coords)
                        best_pt1 = None
                        best_pt2 = None
                        best_len = 0
                        house_min_y = house.bounds[1]
                        is_living = (room["name"] == "living")
                        
                        for k in range(len(coords_seg) - 1):
                            pt1 = coords_seg[k]
                            pt2 = coords_seg[k+1]
                            # For living room, skip the South wall to prevent overlap with the main entrance door
                            living_min_y = living.bounds[1] if living is not None else house.bounds[1]
                            if is_living and abs(pt1[1] - living_min_y) < 0.1 and abs(pt2[1] - living_min_y) < 0.1:
                                continue
                            dx = pt2[0] - pt1[0]
                            dy = pt2[1] - pt1[1]
                            l = (dx*dx + dy*dy) ** 0.5
                            if l > best_len:
                                best_len = l
                                best_pt1 = pt1
                                best_pt2 = pt2
                                
                        if best_pt1 is not None and best_pt2 is not None and best_len > 4.0:
                            mid_x = (best_pt1[0] + best_pt2[0]) / 2
                            mid_y = (best_pt1[1] + best_pt2[1]) / 2
                            
                            dx = best_pt2[0] - best_pt1[0]
                            dy = best_pt2[1] - best_pt1[1]
                            
                            ux = dx / best_len
                            uy = dy / best_len
                            
                            # Window of width 4 centered at (mid_x, mid_y)
                            win_start = (mid_x - 2.0 * ux, mid_y - 2.0 * uy)
                            win_end = (mid_x + 2.0 * ux, mid_y + 2.0 * uy)
                            
                            # Draw window in Dark Blue (color 5) with lineweight 25
                            msp.add_line(
                                win_start, win_end,
                                dxfattribs={"color":5, "lineweight":25}
                            )

        # ==========================================================
        # GATES / DOORS (PRIORITY CONNECTIVITY)
        # ==========================================================

        # Priority list for normal (non-corridor) rooms
        PRIORITY = ["living", "corridor", "kitchen", "bedroom1", "bedroom2", "master", "bedroom3", "bath1", "bath2"]

        def get_priority_rank(name):
            name_lower = name.lower()
            for idx, p in enumerate(PRIORITY):
                if p in name_lower:
                    return idx
            return 999

        drawn_doors = set()

        for room in rooms:
            r_name = room["name"]
            if r_name == "living":
                # Living room is the root space and doesn't need to choose an exit door
                continue

            r_poly = room["polygon"]
            
            # Find all adjacent rooms that touch r_poly
            touching_rooms = []
            for other in rooms:
                o_name = other["name"]
                if o_name == r_name:
                    continue
                o_poly = other["polygon"]
                if r_poly.touches(o_poly):
                    inter = r_poly.intersection(o_poly)
                    if inter.geom_type == "LineString" and inter.length > 4:
                        touching_rooms.append((o_name, o_poly, inter))
            
            # Corridor is a circulation space: draw a door on EVERY shared wall
            if r_name == "corridor":
                for (o_name, o_poly, best_inter) in touching_rooms:
                    door_key = tuple(sorted([r_name, o_name]))
                    if door_key not in drawn_doors:
                        drawn_doors.add(door_key)
                        coords_d = list(best_inter.coords)
                        pt1_d = coords_d[0]
                        pt2_d = coords_d[-1]
                        mid_x = (pt1_d[0] + pt2_d[0]) / 2
                        mid_y = (pt1_d[1] + pt2_d[1]) / 2
                        dx = pt2_d[0] - pt1_d[0]
                        dy = pt2_d[1] - pt1_d[1]
                        length = (dx*dx + dy*dy) ** 0.5
                        if length > 0.001:
                            ux = dx / length
                            uy = dy / length
                            hinge = (mid_x - 1.5 * ux, mid_y - 1.5 * uy)
                            threshold_end = (mid_x + 1.5 * ux, mid_y + 1.5 * uy)
                            msp.add_line(hinge, threshold_end, dxfattribs={"color":1, "lineweight":15})
                            msp.add_line(hinge, threshold_end, dxfattribs={"color":1, "lineweight":30})
                continue

            # All other rooms: pick the single highest-priority adjacent room.
            # Special rule: bath2 always connects to master if adjacent.
            if touching_rooms:
                if r_name == "bath2":
                    master_rooms = [t for t in touching_rooms if t[0] == "master"]
                    if master_rooms:
                        touching_rooms = master_rooms  # force bath2 → master
                touching_rooms.sort(key=lambda item: get_priority_rank(item[0]))
                best_name, best_poly, best_inter = touching_rooms[0]
                
                # Check if we already drew a door between these two rooms
                door_key = tuple(sorted([r_name, best_name]))
                if door_key not in drawn_doors:
                    drawn_doors.add(door_key)
                    
                    coords = list(best_inter.coords)
                    pt1 = coords[0]
                    pt2 = coords[1]

                    mid_x = (pt1[0] + pt2[0]) / 2
                    mid_y = (pt1[1] + pt2[1]) / 2

                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]

                    length = (dx*dx + dy*dy) ** 0.5

                    if length > 0.001:

                        ux = dx / length
                        uy = dy / length

                        # Door hinge (start of opening, centered on wall segment, width 3)
                        hinge = (mid_x - 1.5 * ux, mid_y - 1.5 * uy)
                        threshold_end = (mid_x + 1.5 * ux, mid_y + 1.5 * uy)
                        
                        # Door panel is closed (overlaps with the wall)
                        door_end = threshold_end
                        
                        # Draw the door opening threshold in Red (color 1)
                        msp.add_line(
                            hinge, threshold_end,
                            dxfattribs={"color":1, "lineweight":15}
                        )

                        # Draw the closed door panel in Red (color 1)
                        msp.add_line(
                            hinge, door_end,
                            dxfattribs={"color":1, "lineweight":30}
                        )

        # ==========================================================
        # MAIN ENTRANCE DOOR (SOUTH SIDE OF LIVING ROOM)
        # ==========================================================
        living_room = None
        for room in rooms:
            if room["name"] == "living":
                living_room = room["polygon"]
                break

        if living_room is not None:
            # Use the living room's own south edge (not global house min-Y)
            living_min_y = living_room.bounds[1]
            coords = list(living_room.exterior.coords)
            south_coords = []
            for pt in coords:
                if abs(pt[1] - living_min_y) < 0.1:
                    south_coords.append(pt)
            
            if len(south_coords) >= 2:
                xs = [pt[0] for pt in south_coords]
                min_x_south = min(xs)
                max_x_south = max(xs)
                
                mid_x = (min_x_south + max_x_south) / 2
                
                # Entrance door hinge (centered on south wall, width 3)
                hinge = (mid_x - 1.5, house_min_y)
                threshold_end = (mid_x + 1.5, house_min_y)
                
                # Door panel is closed (overlaps with south wall)
                door_end = threshold_end
                
                # Draw the main entrance door opening threshold in Red (color 1)
                msp.add_line(
                    hinge, threshold_end,
                    dxfattribs={"color":1, "lineweight":15}
                )

                # Draw the main entrance door closed panel in Red (color 1)
                msp.add_line(
                    hinge, door_end,
                    dxfattribs={"color":1, "lineweight":30}
                )

        # ==========================================================
        # TREE
        # ==========================================================

        msp.add_circle(
            TREE_CENTER,
            TREE_RADIUS,
            dxfattribs={"color":5}
        )

        filename = f"house_iteration_{iteration}.dxf"

        doc.saveas(filename)

        return filename
