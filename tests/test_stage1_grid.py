from __future__ import annotations

import math

import pytest

from vacan.simulation.grid import GridSpec


def test_grid_world_round_trip_uses_cell_centers() -> None:
    grid = GridSpec(width_m=12.0, height_m=10.0, resolution_m=0.5)
    for world in [(-5.75, -4.75), (0.25, 0.25), (5.75, 4.75)]:
        row, col = grid.world_to_grid(*world)
        assert grid.grid_to_world(row, col) == world


def test_grid_rejects_outside_coordinates() -> None:
    grid = GridSpec(width_m=4.0, height_m=4.0, resolution_m=0.5)
    with pytest.raises(ValueError):
        grid.world_to_grid(2.0, 0.0)


def test_grid_world_round_trip_returns_cell_center() -> None:
    grid = GridSpec(width_m=8.0, height_m=6.0, resolution_m=0.5)
    row, col = grid.world_to_grid(-1.2, 2.1)
    center_x, center_y = grid.grid_to_world(row, col)
    assert abs(center_x - (-1.2)) <= grid.resolution_m / 2
    assert abs(center_y - 2.1) <= grid.resolution_m / 2


def test_rotated_rectangle_rasterization() -> None:
    grid = GridSpec(width_m=4.0, height_m=4.0, resolution_m=0.5)
    obstacle = {
        "shape": "rectangle",
        "x": 0.0,
        "y": 0.0,
        "length": 2.0,
        "width": 0.5,
        "orientation": math.pi / 2,
    }
    occupancy, _, _ = grid.rasterize([obstacle], [], type("Robot", (), {"x": 1.5, "y": 1.5})())
    assert occupancy.sum() > 3
