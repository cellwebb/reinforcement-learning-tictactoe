WIN_CONDITIONS = (
    # Horizontal
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    # Vertical
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    # Diagonal
    (0, 4, 8),
    (2, 4, 6),
)

# Mapping and inverse mapping for transformations of the board
TRANSFORMATIONS = (
    ("original", tuple(range(9)), tuple(range(9))),
    ("rotate_90", (6, 3, 0, 7, 4, 1, 8, 5, 2), (2, 5, 8, 1, 4, 7, 0, 3, 6)),
    ("rotate_180", (8, 7, 6, 5, 4, 3, 2, 1, 0), (8, 7, 6, 5, 4, 3, 2, 1, 0)),
    ("rotate_270", (2, 5, 8, 1, 4, 7, 0, 3, 6), (6, 3, 0, 7, 4, 1, 8, 5, 2)),
    ("reflect_horizontal", (2, 1, 0, 5, 4, 3, 8, 7, 6), (2, 1, 0, 5, 4, 3, 8, 7, 6)),
    ("reflect_horizontal_rotate_90", (8, 5, 2, 7, 4, 1, 6, 3, 0), (8, 5, 2, 7, 4, 1, 6, 3, 0)),
    ("reflect_horizontal_rotate_180", (6, 7, 8, 3, 4, 5, 0, 1, 2), (6, 7, 8, 3, 4, 5, 0, 1, 2)),
    ("reflect_horizontal_rotate_270", (0, 3, 6, 1, 4, 7, 2, 5, 8), (0, 3, 6, 1, 4, 7, 2, 5, 8)),
)

INVERSE_MAPPINGS = {
    "original": tuple(range(9)),
    "rotate_90": (2, 5, 8, 1, 4, 7, 0, 3, 6),
    "rotate_180": (8, 7, 6, 5, 4, 3, 2, 1, 0),
    "rotate_270": (6, 3, 0, 7, 4, 1, 8, 5, 2),
    "reflect_horizontal": (2, 1, 0, 5, 4, 3, 8, 7, 6),
    "reflect_horizontal_rotate_90": (6, 3, 0, 7, 4, 1, 8, 5, 2),
    "reflect_horizontal_rotate_180": (8, 7, 6, 5, 4, 3, 2, 1, 0),
    "reflect_horizontal_rotate_270": (0, 3, 6, 1, 4, 7, 2, 5, 8),
}
