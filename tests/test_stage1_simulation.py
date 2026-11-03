from __future__ import annotations

from pathlib import Path

import pytest

from vacan.simulation.stage1 import density_grid, episode_seed, grid_spec, load_stage1_config

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "office": ROOT / "configs/scenarios/office_lobby.yaml",
    "hospital": ROOT / "configs/scenarios/hospital_corridor.yaml",
    "warehouse": ROOT / "configs/scenarios/warehouse.yaml",
}


@pytest.mark.parametrize(("name", "path"), SCENARIOS.items())
def test_scenario_uses_official_isaac_pegasus_contract(name: str, path: Path) -> None:
    config = load_stage1_config(path)
    assert config["scenario"]["name"] == name
    assert config["backend"] == "isaac-sim-pegasus"
    assert config["environment"]["asset"].startswith("Isaac/Environments/")
    assert len(config["pedestrians"]) >= 2
    assert all(len(person["destinations"]) >= 2 for person in config["pedestrians"])


@pytest.mark.parametrize("path", SCENARIOS.values())
def test_dataset_splits_have_independent_seeds(path: Path) -> None:
    config = load_stage1_config(path)
    train = {episode_seed(config, "train", i) for i in range(len(config["seeds"]["train"]))}
    validation = set(config["seeds"]["validation"])
    test = set(config["seeds"]["test"])
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_exact_density_grid_uses_world_frame_cell_area() -> None:
    config = load_stage1_config(SCENARIOS["office"])
    spec = grid_spec(config)
    counts, density = density_grid([(-11.75, -11.75), (-11.6, -11.6), (0.0, 0.0)], spec)
    assert sum(map(sum, counts)) == 3
    assert counts[0][0] == 2
    assert density[0][0] == 2 / spec.resolution_m**2


def test_ground_truth_grid_excludes_out_of_bounds_people() -> None:
    config = load_stage1_config(SCENARIOS["warehouse"])
    counts, _ = density_grid([(1000.0, 1000.0)], grid_spec(config))
    assert sum(map(sum, counts)) == 0
