WIN_CONDITIONS = [
    [0, 1, 2],  # Horizontal
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],  # Vertical
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],  # Diagonal
    [2, 4, 6],
]

TRANSFORMATIONS = [
    ("original", None),
    ("rotate_90", [6, 3, 0, 7, 4, 1, 8, 5, 2]),
    ("rotate_180", [8, 7, 6, 5, 4, 3, 2, 1, 0]),
    ("rotate_270", [2, 5, 8, 1, 4, 7, 0, 3, 6]),
    ("reflect_horizontal", [2, 1, 0, 5, 4, 3, 8, 7, 6]),
    ("reflect_horizontal_rotate_90", [8, 5, 2, 7, 4, 1, 6, 3, 0]),
    ("reflect_horizontal_rotate_180", [6, 7, 8, 3, 4, 5, 0, 1, 2]),
    ("reflect_horizontal_rotate_270", [0, 3, 6, 1, 4, 7, 2, 5, 8]),
]

INVERSE_MAPPINGS = {
    "rotate_90": [2, 5, 8, 1, 4, 7, 0, 3, 6],
    "rotate_180": [8, 7, 6, 5, 4, 3, 2, 1, 0],
    "rotate_270": [6, 3, 0, 7, 4, 1, 8, 5, 2],
    "reflect_horizontal": [2, 1, 0, 5, 4, 3, 8, 7, 6],
    "reflect_horizontal_rotate_90": [6, 3, 0, 7, 4, 1, 8, 5, 2],
    "reflect_horizontal_rotate_180": [8, 7, 6, 5, 4, 3, 2, 1, 0],
    "reflect_horizontal_rotate_270": [2, 5, 8, 1, 4, 7, 0, 3, 6],
}
