from dataclasses import dataclass

# ============================================================================
# CONFIG
# ============================================================================

PLOT_COORDS = [
    (0, 8),
    (35, 8),
    (35, 0),
    (75, 0),
    (75, 8),
    (80, 8),
    (80, 88),
    (75, 88),
    (75, 84),
    (0, 84)
]

TREE_CENTER = (75, 88)

TREE_DIAMETER = 24
TREE_RADIUS = TREE_DIAMETER / 2

FRONT_SETBACK = 20
REAR_SETBACK = 15
LEFT_SETBACK = 5
RIGHT_SETBACK = 5

MAX_ITERATIONS = 10

MAX_X = max(x for x, y in PLOT_COORDS)
MAX_Y = max(y for x, y in PLOT_COORDS)

# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class LayoutProposal:

    split_x: float
    right_height: float
