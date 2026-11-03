from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from vacan.checkpoints import CheckpointManager
from vacan.config import load_config
from vacan.seeding import seed_everything
from vacan.simulation import DynamicPedestrianSimulation


ROOT = Path(__file__).resolve().parents[1]


def test_central_config_preserves_research_factors() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    assert config["project"]["research_factors"] == ["congestion", "intention", "freshness"]
    assert config["simulation"]["backend"] == "isaac-sim"
    assert config["simulation"]["isaac_sim_version"] == "5.1.0"
    assert config["simulation"]["isaac_lab_version"] == "2.3.2"
    assert config["simulation"]["pegasus_version"] == "5.1.0"
    assert config["simulation"]["robot"] == "nova-carter"
    assert config["simulation"]["privileged_evaluation"] is False
    assert config["simulation"]["ground_truth_policy_access"] is False
    assert config["hardware"]["simulation_gpu_index"] == 1
    assert config["hardware"]["headless"] is False
    assert config["hardware"]["desktop_only_gpu_indices"] == [2]
    assert config["hardware"]["xorg_gpu_pci_bus_id"] == "PCI:115:0:0"


def test_stage0_smoke_config_uses_verified_assets() -> None:
    config = load_config(ROOT / "configs/isaacsim/stage0_smoke.yaml")
    assets = config["assets"]
    assert assets["environments"] == {
        "office": "/Isaac/Environments/Office/office.usd",
        "hospital": "/Isaac/Environments/Hospital/hospital.usd",
        "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    }
    assert assets["nova_carter"] == "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"


def test_required_stage0_package_boundaries_exist() -> None:
    for package in (
        "simulation",
        "pedestrians",
        "robot",
        "sensors",
        "mapping",
        "prediction",
        "rl",
        "reference",
        "counterfactual",
        "analysis",
    ):
        assert (ROOT / "src/vacan" / package / "__init__.py").is_file()


def test_seed_everything_repeats_python_and_numpy() -> None:
    seed_everything(17)
    first = (random.random(), np.random.random())
    seed_everything(17)
    second = (random.random(), np.random.random())
    assert first == second


def test_checkpoint_manager_writes_metadata_and_prunes(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=1)

    def writer(path: Path) -> None:
        path.write_bytes(b"weights")

    manager.save("run", 1, 17, writer)
    latest = manager.save("run", 2, 17, writer)
    assert latest.read_bytes() == b"weights"
    assert not (tmp_path / "run-step-000000001.ckpt").exists()
    metadata = json.loads(latest.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 17
    assert metadata["step"] == 2


def test_policy_adapter_does_not_expose_privileged_snapshot() -> None:
    assert not hasattr(DynamicPedestrianSimulation, "snapshot")
    assert not hasattr(DynamicPedestrianSimulation, "ground_truth")
