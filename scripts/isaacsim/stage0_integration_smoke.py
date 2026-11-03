#!/usr/bin/env python3
"""Isaac Sim + Isaac Lab + Pegasus + Nova Carter Stage 0 smoke test."""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
ARGS = parser.parse_args()
CONFIG = yaml.safe_load(ARGS.config.resolve().read_text(encoding="utf-8"))

# Follow Pegasus' standalone launch path. Its 5.1 installer adds
# isaacsim.replicator.agent.core to Isaac Sim's base experience, which pulls
# the matching People, navigation, and animation dependencies. Loading the UI
# bundles again here changes extension ordering and can break AnimationGraph
# registration, so do not add a second dependency path.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": bool(ARGS.headless),
        "active_gpu": int(ARGS.gpu),
        "physics_gpu": int(ARGS.gpu),
        "multi_gpu": False,
        "max_gpu_count": 1,
        "sync_loads": True,
        "fast_shutdown": True,
    }
)

# Isaac/Kit imports must follow application startup.
import numpy as np
import omni.client
import omni.timeline
import omni.usd
import AnimGraphSchema
import omni.anim.graph.core as animation_graph_core
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaacsim.core.api.world import World
from isaacsim.storage.native import get_assets_root_path

# Pegasus' official people example requires a fresh stage after the People
# dependencies have loaded. Without it, ApplyAnimationGraphAPICommand can act
# on the previous startup stage while World and Person retain another stage.
if not omni.usd.get_context().new_stage():
    raise RuntimeError("Isaac Sim could not create the fresh Pegasus People stage")
simulation_app.update()

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController


class FixedTargetController(PersonController):
    def __init__(self, target: list[float], speed: float) -> None:
        super().__init__()
        self.target = target
        self.speed = speed

    def update(self, dt: float) -> None:
        del dt
        self._person.update_target_position(self.target, walk_speed=self.speed)


def asset_url(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{relative.lstrip('/')}"


def require_asset(url: str) -> None:
    result, _ = omni.client.stat(url)
    if result != omni.client.Result.OK:
        raise FileNotFoundError(f"Isaac asset is unavailable: {url} ({result})")


def world_translation(stage: Usd.Stage, prim_path: str) -> list[float]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim was not spawned: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    value = matrix.ExtractTranslation()
    return [float(value[0]), float(value[1]), float(value[2])]


def write_result(payload: dict[str, object]) -> None:
    ARGS.result.parent.mkdir(parents=True, exist_ok=True)
    ARGS.result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_person_animation(stage: Usd.Stage, person: Person) -> dict[str, object]:
    """Fail before stepping if Pegasus did not wire the animation graph."""

    current_stage = omni.usd.get_context().get_stage()
    if stage is not current_stage:
        raise RuntimeError("Pegasus World stage is not the active omni.usd stage")

    skel_root = person.character_skel_root
    skel_path = str(person.character_skel_root_stage_path)
    graph_path = f"{Person.character_root_prim_path}/Biped_Setup/CharacterAnimation/AnimationGraph"
    graph_prim = stage.GetPrimAtPath(graph_path)
    diagnostics = {
        "skel_root_path": skel_path,
        "skel_root_type": skel_root.GetTypeName() if skel_root and skel_root.IsValid() else None,
        "animation_graph_path": graph_path,
        "animation_graph_valid": bool(graph_prim and graph_prim.IsValid()),
        "animation_graph_api_applied": bool(
            skel_root
            and skel_root.IsValid()
            and skel_root.HasAPI(AnimGraphSchema.AnimationGraphAPI)
        ),
    }
    if diagnostics["skel_root_type"] != "SkelRoot":
        raise RuntimeError(f"Pegasus character has no valid SkelRoot: {diagnostics}")
    if not diagnostics["animation_graph_valid"]:
        raise RuntimeError(f"Pegasus Biped animation graph did not load: {diagnostics}")
    if not diagnostics["animation_graph_api_applied"]:
        raise RuntimeError(f"Pegasus failed to apply AnimationGraphAPI: {diagnostics}")
    return diagnostics


def run() -> dict[str, object]:
    started = time.perf_counter()
    sim_cfg = CONFIG["simulation"]
    scene_cfg = CONFIG["scene"]
    seed = int(CONFIG["reproducibility"]["seed"])
    random.seed(seed)
    np.random.seed(seed)

    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim assets root could not be resolved")
    environment_key = str(sim_cfg["environment"])
    environment_url = asset_url(root, CONFIG["assets"]["environments"][environment_key])
    robot_url = asset_url(root, CONFIG["assets"]["nova_carter"])
    require_asset(environment_url)
    require_asset(robot_url)

    pg = PegasusInterface()
    pg._world_settings = {
        "physics_dt": float(sim_cfg["physics_dt"]),
        "rendering_dt": float(sim_cfg["rendering_dt"]),
        "stage_units_in_meters": 1.0,
        "device": "cpu",
    }
    pg._world = World(**pg._world_settings)
    world = pg.world
    if world.stage is not omni.usd.get_context().get_stage():
        raise RuntimeError("World was created on a stage other than the active Pegasus stage")

    environment_spawn = sim_utils.UsdFileCfg(usd_path=environment_url)
    environment_spawn.func(scene_cfg["environment_prim"], environment_spawn)
    robot_spawn = sim_utils.UsdFileCfg(usd_path=robot_url)
    robot_spawn.func(
        scene_cfg["robot_prim"], robot_spawn, translation=tuple(scene_cfg["robot_position"])
    )

    controller = FixedTargetController(
        target=list(scene_cfg["person_target"]), speed=float(scene_cfg["person_speed_m_s"])
    )
    person = Person(
        str(scene_cfg["person_name"]),
        str(CONFIG["assets"]["pedestrian"]["character"]),
        init_pos=list(scene_cfg["person_start"]),
        controller=controller,
    )

    # Allow referenced character and Biped assets to finish composing, then
    # verify the exact API required by PersonController before an expensive
    # simulation loop is started.
    for _ in range(3):
        simulation_app.update()
    animation = validate_person_animation(world.stage, person)

    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    stage = world.stage
    animation_after_reset = validate_person_animation(stage, person)

    # Pegasus creates the live character handle only after the timeline has
    # started and several application/render updates have synchronized USD,
    # Animation Graph, and Fabric. This warm-up is required independently of
    # whether the application window is enabled.
    for _ in range(5):
        world.step(render=True)
    person.character_graph = animation_graph_core.get_character(
        person.character_skel_root_stage_path
    )
    if person.character_graph is None:
        raise RuntimeError(
            "Pegasus AnimationGraphAPI exists but no live character handle was created "
            f"after reset/render synchronization: {animation_after_reset}"
        )

    robot_initial = world_translation(stage, scene_cfg["robot_prim"])
    person_initial = np.asarray(person.state.position, dtype=float).copy()
    steps = int(sim_cfg["steps"])
    for _ in range(steps):
        if not simulation_app.is_running():
            raise RuntimeError("SimulationApp stopped before requested steps completed")
        world.step(render=True)
    robot_final = world_translation(stage, scene_cfg["robot_prim"])
    person_final = np.asarray(person.state.position, dtype=float).copy()
    movement = float(np.linalg.norm(person_final[:2] - person_initial[:2]))
    threshold = float(sim_cfg["movement_threshold_m"])
    if movement < threshold:
        raise AssertionError(f"Pegasus person moved {movement:.6f} m; expected >= {threshold:.6f} m")
    timeline.stop()

    return {
        "status": "pass",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "runtime": {
            "gpu_index": int(ARGS.gpu),
            "headless": bool(ARGS.headless),
        },
        "versions_expected": {
            "isaac_sim": sim_cfg["isaac_sim_version"],
            "isaac_lab": sim_cfg["isaac_lab_version"],
            "pegasus": sim_cfg["pegasus_version"],
        },
        "assets": {"root": root, "environment": environment_url, "nova_carter": robot_url},
        "steps": steps,
        "robot": {"initial": robot_initial, "final": robot_final},
        "person": {
            "initial": person_initial.tolist(),
            "final": person_final.tolist(),
            "movement_m": movement,
            "animation": animation,
        },
        "elapsed_s": time.perf_counter() - started,
    }


if __name__ == "__main__":
    result: dict[str, object] = {}
    exit_code = 0
    try:
        result = run()
    except Exception as error:
        exit_code = 1
        result = {
            "status": "fail",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            try:
                omni.timeline.get_timeline_interface().stop()
            finally:
                write_result(result)
        finally:
            simulation_app.close()
    raise SystemExit(exit_code)
