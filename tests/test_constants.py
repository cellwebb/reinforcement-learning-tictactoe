from tictactoe.constants import TRANSFORMATIONS


def test_TRANSFORMATIONS():
    """Test the TRANSFORMATIONS constant."""
    assert len(TRANSFORMATIONS) == 8
    assert all(len(t) == 3 for t in TRANSFORMATIONS)
    for name, forward, backward in TRANSFORMATIONS:
        assert name in [
            "original",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "reflect_horizontal",
            "reflect_horizontal_rotate_90",
            "reflect_horizontal_rotate_180",
            "reflect_horizontal_rotate_270",
        ]
        assert len(forward) == 9
        assert len(backward) == 9
        assert len(set(forward)) == 9
        assert len(set(backward)) == 9
        assert all(0 <= i < 9 for i in forward)
        assert all(0 <= i < 9 for i in backward)
        for i in range(9):
            assert forward[i] == backward.index(i), f"Error in {name}, {i}, {forward}, {backward}"
            assert backward[i] == forward.index(i), f"Error in {name}, {i}, {forward}, {backward}"
