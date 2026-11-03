#!/usr/bin/env python3
"""Run one deterministic VACaN Stage 1 episode in Isaac Sim with Pegasus People."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
parser.add_argument("--episode-index", type=int, required=True)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int)
parser.add_argument("--route-cache-input", type=Path)
ARGS = parser.parse_args()
CONFIG = yaml.safe_load(ARGS.config.resolve().read_text(encoding="utf-8"))

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": bool(ARGS.headless), "active_gpu": int(ARGS.gpu),
        "physics_gpu": int(ARGS.gpu), "multi_gpu": False, "max_gpu_count": 1,
        "sync_loads": True, "fast_shutdown": True,
    }
)

# Kit/Isaac imports must happen after SimulationApp startup.
import carb
import numpy as np
import omni.anim.graph.core as animation_graph_core
import omni.anim.navigation.core as navigation
import omni.client
import omni.timeline
import omni.usd
import NavSchema
from pxr import Gf, Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaacsim.core.api.world import World
from isaacsim.storage.native import get_assets_root_path
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from vacan.simulation.stage1 import density_grid, grid_spec, load_stage1_config


def vector(value) -> np.ndarray:
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def closest_navmesh_point(navmesh, point: np.ndarray, radius: float) -> np.ndarray:
    result = navmesh.query_closest_point(carb.Float3(*point.tolist()), agent_radius=radius)
    if not result:
        raise RuntimeError(f"NavMesh has no closest point for {point.tolist()}")
    return vector(result[0])


def navmesh_path_points(navmesh, start: np.ndarray, end: np.ndarray, radius: float) -> list[np.ndarray]:
    path = navmesh.query_shortest_path(
        start_pos=carb.Float3(*start.tolist()),
        end_pos=carb.Float3(*end.tolist()),
        agent_radius=radius,
    )
    if path is None or not path.get_points():
        raise RuntimeError(f"No NavMesh path: {start.tolist()} -> {end.tolist()}")
    return [vector(point) for point in path.get_points()]


def wait_for_asset_composition(timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    context = omni.usd.get_context()
    while context.get_stage_loading_status()[2] > 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("USD asset composition exceeded timeout")
        simulation_app.update()


def require_remote_asset(path: str, timeout_s: float = 30.0) -> None:
    """Wait for OmniClient to resolve an official remote asset path."""

    deadline = time.monotonic() + timeout_s
    result = None
    while time.monotonic() < deadline:
        result, _ = omni.client.stat(path)
        if result == omni.client.Result.OK:
            return
        for _ in range(3):
            simulation_app.update()
        time.sleep(0.25)
    raise FileNotFoundError(f"Official remote asset unavailable: {path} ({result})")


def bake_navmesh(stage, cfg: dict):
    value = cfg["navmesh"]
    center, size = vector(value["volume_center"]), vector(value["volume_size"])
    volume = NavSchema.NavMeshVolume.Define(stage, "/World/VACaNNavMeshVolume")
    volume.GetNavVolumeTypeAttr().Set("Include")
    prim = volume.GetPrim()
    prim.ApplyAPI(NavSchema.NavMeshAreaAPI)
    UsdGeom.Boundable(prim).GetExtentAttr().Set(
        [Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)]
    )
    prim.CreateAttribute("nav:area", Sdf.ValueTypeNames.String).Set("Walkable")
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
    xform.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))
    if not prim.IsValid() or not prim.IsA(NavSchema.NavMeshVolume):
        raise RuntimeError("Authored NavMeshVolume is invalid")
    if stage is not omni.usd.get_context().get_stage():
        raise RuntimeError("NavMeshVolume was authored on a non-active stage")
    # INavigation observes the live USD/Fabric stage, not unprocessed authoring
    # changes. Flush the new schema prim before requesting a bake.
    for _ in range(3):
        simulation_app.update()
    settings = carb.settings.get_settings()
    settings.set(navigation.NavMeshSettings.AUTO_REBAKE_SETTING_PATH, False)
    # Reuse the extension's content-addressed bake cache across clean episode
    # processes. This removes avoidable tessellation/path-branch variation for
    # identical official USD assets and bake settings.
    settings.set(navigation.NavMeshSettings.CACHE_ENABLED_SETTING_PATH, True)
    # Isaac Sim 5.1's navigation extension stores bake preferences in cm even
    # when the USD stage uses meters. Query points/radii remain in stage units.
    cm_per_m = 100.0
    settings.set(navigation.NavMeshSettings.AGENT_MIN_HEIGHT_SETTING_PATH, float(value["agent_min_height_m"]) * cm_per_m)
    settings.set(navigation.NavMeshSettings.AGENT_MIN_RADIUS_SETTING_PATH, float(value["agent_radius_m"]) * 0.75 * cm_per_m)
    settings.set(navigation.NavMeshSettings.AGENT_MAX_RADIUS_SETTING_PATH, float(value["agent_radius_m"]) * cm_per_m)
    settings.set(navigation.NavMeshSettings.AGENT_MAX_STEP_HEIGHT_SETTING_PATH, float(value["agent_max_step_height_m"]) * cm_per_m)
    settings.set(navigation.NavMeshSettings.AGENT_MAX_FLOOR_SLOPE_SETTING_PATH, float(value["agent_max_floor_slope_deg"]))
    settings.set(navigation.NavMeshSettings.AGENT_MIN_ISLAND_RADIUS_SETTING_PATH, float(value["min_island_radius_m"]) * cm_per_m)
    interface = navigation.acquire_interface()
    payloads: list[object] = []
    # The 5.1 navigation extension can emit a completed event with no mesh
    # while a large referenced environment is still becoming available to its
    # geometry collector. Retry only that observed transient failure; every
    # attempt remains bounded and a persistent failure is fatal.
    for attempt in range(1, 4):
        state: dict[str, object] = {"done": False, "payload": None}

        def on_event(event) -> None:
            if event.type == navigation.EVENT_TYPE_NAVMESH_UPDATED:
                state["done"], state["payload"] = True, event.payload.get_dict()

        subscription = interface.get_navmesh_event_stream().create_subscription_to_pop(
            on_event, name=f"VACaN NavMesh bake attempt {attempt}"
        )
        interface.start_navmesh_baking()
        deadline = time.monotonic() + 300.0
        while not state["done"]:
            if time.monotonic() >= deadline:
                del subscription
                raise TimeoutError(f"NavMesh baking attempt {attempt} exceeded timeout")
            simulation_app.update()
        del subscription
        navmesh = interface.get_navmesh()
        if navmesh is not None:
            return interface, navmesh
        payloads.append(state["payload"])
        for _ in range(30):
            simulation_app.update()
    raise RuntimeError(f"NavMesh bake failed after 3 attempts: {payloads}")


class NavMeshCrowdController(PersonController):
    """Pegasus controller adding NavMesh routing and deterministic yielding.

    Pegasus 5.1 provides no SFM, ORCA, obstacle avoidance, or social behavior
    in PersonController. Static avoidance here comes from the official scene's
    baked NavMesh; priority yielding is explicitly not SFM or ORCA.
    """

    instances: list["NavMeshCrowdController"] = []

    def __init__(self, pedestrian_id, destinations, speed_m_s, navmesh, config,
                 agent_radius_m, route_offset, initial_path, cycle_paths) -> None:
        super().__init__()
        self.pedestrian_id = pedestrian_id
        self.destinations = destinations
        self.speed_m_s = float(speed_m_s)
        self.navmesh = navmesh
        self.config = config
        self.agent_radius_m = agent_radius_m
        self.destination_index = route_offset % len(destinations)
        self.initial_path = [point.copy() for point in initial_path]
        self.cycle_paths = {
            int(index): [point.copy() for point in points]
            for index, points in cycle_paths.items()
        }
        self.used_initial_path = False
        self.path_points: list[np.ndarray] = []
        self.commanded_target: np.ndarray | None = None
        self.deterministic_position: np.ndarray | None = None
        self.previous_deterministic_position: np.ndarray | None = None
        self.replans = 0
        self.instances.append(self)

    @property
    def current_destination(self) -> np.ndarray:
        return self.destinations[self.destination_index]

    def _plan(self, position: np.ndarray) -> None:
        if not self.used_initial_path:
            points = self.initial_path
            self.used_initial_path = True
        else:
            points = self.cycle_paths[self.destination_index]
        if not points:
            raise RuntimeError(f"Empty resolved route for {self.pedestrian_id}")
        self.path_points = points[1:] if len(points) > 1 else points
        self.replans += 1

    def _interaction_speed_scale(self, position: np.ndarray) -> float:
        """Yield to a stable ID priority without changing the selected route."""

        radius = float(self.config["separation_radius_m"])
        scale = 1.0
        quantization_m = 0.1
        position_xy = np.round(position[:2] / quantization_m) * quantization_m
        for other in self.instances:
            if other is self or other.person is None:
                continue
            other_position = (
                other.deterministic_position
                if other.deterministic_position is not None
                else vector(other.person.state.position)
            )
            other_xy = np.round(other_position[:2] / quantization_m) * quantization_m
            distance = float(np.linalg.norm(position_xy - other_xy))
            if distance < radius and self.pedestrian_id > other.pedestrian_id:
                scale = min(scale, 0.5)
        return scale

    def update(self, dt: float) -> None:
        if self.deterministic_position is None:
            self.deterministic_position = vector(self._person.state.position)
            self.previous_deterministic_position = self.deterministic_position.copy()
        position = self.deterministic_position
        if not self.path_points:
            self._plan(position)
        tolerance = float(self.config["waypoint_tolerance_m"])
        while self.path_points and np.linalg.norm(self.path_points[0][:2] - position[:2]) <= tolerance:
            self.path_points.pop(0)
        if not self.path_points:
            self.destination_index = (self.destination_index + 1) % len(self.destinations)
            self._plan(position)
        target = self.path_points[0].copy()
        delta = target - position
        distance = float(np.linalg.norm(delta[:2]))
        lookahead = float(self.config["lookahead_m"])
        if distance > lookahead:
            target = position + delta / distance * lookahead
        self.commanded_target = target
        direction = target - position
        planar_distance = float(np.linalg.norm(direction[:2]))
        if planar_distance > 1e-9:
            interaction_scale = self._interaction_speed_scale(position)
            displacement = min(
                self.speed_m_s * interaction_scale * float(dt), planar_distance
            )
            next_position = position + direction / planar_distance * displacement
            self.previous_deterministic_position = position.copy()
            self.deterministic_position = next_position
        self._person.update_target_position(target.tolist(), walk_speed=self.speed_m_s)

    def enforce_deterministic_pose(self) -> None:
        """Remove animation root-motion drift while retaining Pegasus animation."""

        if self.deterministic_position is None or self._person.character_graph is None:
            return
        previous = self.previous_deterministic_position
        delta = self.deterministic_position - previous
        if float(np.linalg.norm(delta[:2])) > 1e-9:
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            orientation = np.asarray(
                [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=np.float64
            )
        else:
            orientation = np.asarray(self._person.state.orientation, dtype=np.float64)
            if orientation.shape[0] != 4:
                orientation = np.asarray([0.0, 0.0, 0.0, 1.0])
        self._person.character_graph.set_world_transform(
            carb.Float3(*self.deterministic_position.tolist()), carb.Float4(*orientation.tolist())
        )
        self._person.state.position = self.deterministic_position.copy()
        self._person.state.orientation = orientation.copy()

    def reset(self) -> None:
        self.path_points.clear()
        self.used_initial_path = False
        self.deterministic_position = None
        self.previous_deterministic_position = None


class GroundTruthWriter:
    def __init__(self, path: Path, config: dict, seed: int, navmesh) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.stream = path, path.open("w", encoding="utf-8")
        self.config, self.seed, self.navmesh = config, seed, navmesh
        self.grid = grid_spec(config)
        self.previous: dict[str, np.ndarray] = {}
        self.max_navmesh_distance_m = 0.0
        self.zone_ranges = {
            name: [math.inf, -math.inf]
            for name in config["ground_truth"]["congestion_zones"]
        }

    def write(self, step, dt, people, asset_url) -> None:
        sim_time = step * dt
        start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.seed)
        pedestrians, positions = [], []
        for cfg, person, controller in people:
            position = vector(person.state.position)
            previous = self.previous.get(cfg["id"], position)
            velocity = (position - previous) / dt if step > 0 else np.zeros(3)
            self.previous[cfg["id"]] = position.copy()
            closest = closest_navmesh_point(
                self.navmesh, position, float(self.config["navmesh"]["agent_radius_m"])
            )
            navmesh_distance = float(np.linalg.norm(position - closest))
            self.max_navmesh_distance_m = max(self.max_navmesh_distance_m, navmesh_distance)
            orientation = getattr(person.state, "orientation", person.state.attitude)
            pedestrians.append({
                "id": cfg["id"], "prim_path": person._stage_prefix,
                "position_xyz_m": position.tolist(),
                "orientation_xyzw": [float(v) for v in orientation],
                "velocity_xyz_m_s": velocity.tolist(),
                "current_destination_xyz_m": controller.current_destination.tolist(),
                "commanded_target_xyz_m": (
                    controller.commanded_target.tolist()
                    if controller.commanded_target is not None else position.tolist()
                ),
                "speed_setting_m_s": controller.speed_m_s,
                "navmesh_distance_m": navmesh_distance,
            })
            positions.append((float(position[0]), float(position[1])))
        counts, density = density_grid(positions, self.grid)
        zones = {}
        for name, bounds in self.config["ground_truth"]["congestion_zones"].items():
            x0, y0, x1, y1 = (float(v) for v in bounds)
            count = sum(x0 <= x <= x1 and y0 <= y <= y1 for x, y in positions)
            zones[name] = {"bounds_xy_m": [x0, y0, x1, y1], "pedestrian_count": count}
            self.zone_ranges[name][0] = min(self.zone_ranges[name][0], count)
            self.zone_ranges[name][1] = max(self.zone_ranges[name][1], count)
        payload = {
            "timestamp": (start + timedelta(seconds=sim_time)).isoformat(),
            "simulation_step": step, "simulation_time_s": sim_time, "seed": self.seed,
            "world_frame": self.config["ground_truth"]["world_frame"],
            "environment_state": {
                "robot_independent": True, "official_isaac_asset": asset_url,
                "scenario": self.config["scenario"]["name"],
            },
            "pedestrians": pedestrians,
            "ground_truth_density_grid": {
                "origin_xy_m": list(self.grid.origin_xy), "resolution_m": self.grid.resolution_m,
                "shape": list(self.grid.shape), "counts": counts, "people_per_m2": density,
            },
            "congestion_zones": zones,
        }
        self.stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    def close(self) -> None:
        self.stream.close()


def run() -> dict[str, object]:
    started = time.perf_counter()
    config = load_stage1_config(ARGS.config.resolve())
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    NavMeshCrowdController.instances.clear()
    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim official asset root is unavailable")
    asset_url = f"{root.rstrip('/')}/{config['environment']['asset'].lstrip('/')}"
    result, _ = omni.client.stat(asset_url)
    if result != omni.client.Result.OK:
        raise FileNotFoundError(f"Official Isaac asset unavailable: {asset_url} ({result})")
    if not omni.usd.get_context().new_stage():
        raise RuntimeError("Could not create a fresh Pegasus People stage")
    simulation_app.update()
    episode = config["episode"]
    pg = PegasusInterface()
    pg._world_settings = {
        "physics_dt": float(episode["physics_dt_s"]),
        "rendering_dt": float(episode["rendering_dt_s"]),
        "stage_units_in_meters": 1.0, "device": "cpu",
    }
    pg._world = World(**pg._world_settings)
    world = pg.world
    environment_spawn = sim_utils.UsdFileCfg(usd_path=asset_url)
    environment_spawn.func(config["environment"]["prim_path"], environment_spawn)
    wait_for_asset_composition()
    navigation_interface, navmesh = bake_navmesh(world.stage, config)
    navigation_interface.set_random_seed(config["scenario"]["name"], ARGS.seed)
    agent_radius = float(config["navmesh"]["agent_radius_m"])
    people, spawn_snaps = [], {}
    route_rng = random.Random(ARGS.seed)
    output = ARGS.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cached_routes = None
    if ARGS.route_cache_input is not None:
        cached_routes = json.loads(ARGS.route_cache_input.resolve().read_text(encoding="utf-8"))
        if (cached_routes.get("scenario") != config["scenario"]["name"] or
                int(cached_routes.get("seed", -1)) != ARGS.seed):
            raise ValueError("Resolved route cache scenario/seed mismatch")
    resolved_routes: dict[str, object] = {
        "schema_version": 1, "scenario": config["scenario"]["name"],
        "seed": ARGS.seed, "agent_radius_m": agent_radius, "pedestrians": {},
    }
    for character in sorted({person["character"] for person in config["pedestrians"]}):
        require_remote_asset(f"{Person.assets_root_path.rstrip('/')}/{character}")
    for pedestrian in config["pedestrians"]:
        requested_spawn = vector(pedestrian["spawn"])
        cached = None if cached_routes is None else cached_routes["pedestrians"][pedestrian["id"]]
        spawn = (
            closest_navmesh_point(navmesh, requested_spawn, agent_radius)
            if cached is None else vector(cached["spawn_xyz_m"])
        )
        snap_distance = float(np.linalg.norm(spawn - requested_spawn))
        spawn_snaps[pedestrian["id"]] = snap_distance
        maximum = float(config["navmesh"]["max_spawn_snap_m"])
        if snap_distance > maximum:
            raise AssertionError(
                f"{pedestrian['id']} spawn snap {snap_distance:.3f} m exceeds {maximum:.3f} m"
            )
        if cached is None:
            destinations = [
                closest_navmesh_point(navmesh, vector(point), agent_radius)
                for point in pedestrian["destinations"]
            ]
            route_offset = route_rng.randrange(len(destinations))
            initial_path = navmesh_path_points(
                navmesh, spawn, destinations[route_offset], agent_radius
            )
            cycle_paths = {
                index: navmesh_path_points(
                    navmesh, destinations[(index - 1) % len(destinations)],
                    destination, agent_radius,
                )
                for index, destination in enumerate(destinations)
            }
        else:
            destinations = [vector(point) for point in cached["destinations_xyz_m"]]
            route_offset = int(cached["route_offset"])
            initial_path = [vector(point) for point in cached["initial_path_xyz_m"]]
            cycle_paths = {
                int(index): [vector(point) for point in points]
                for index, points in cached["cycle_paths_xyz_m"].items()
            }
        resolved_routes["pedestrians"][pedestrian["id"]] = {
            "spawn_xyz_m": spawn.tolist(),
            "destinations_xyz_m": [point.tolist() for point in destinations],
            "route_offset": route_offset,
            "initial_path_xyz_m": [point.tolist() for point in initial_path],
            "cycle_paths_xyz_m": {
                str(index): [point.tolist() for point in points]
                for index, points in cycle_paths.items()
            },
        }
        controller = NavMeshCrowdController(
            pedestrian["id"], destinations, pedestrian["speed_m_s"], navmesh,
            config["pedestrian_controller"], agent_radius,
            route_offset, initial_path, cycle_paths,
        )
        person = Person(
            pedestrian["id"], pedestrian["character"],
            init_pos=spawn.tolist(), controller=controller,
        )
        people.append((pedestrian, person, controller))
    (output / "resolved_routes.json").write_text(
        json.dumps(resolved_routes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for _ in range(3):
        simulation_app.update()
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(int(episode["warmup_steps"])):
        world.step(render=True)
        for _, _, controller in people:
            controller.enforce_deterministic_pose()
    for _, person, _ in people:
        person.character_graph = animation_graph_core.get_character(person.character_skel_root_stage_path)
        if person.character_graph is None:
            raise RuntimeError(f"No live Pegasus animation graph for {person._stage_prefix}")
    writer = GroundTruthWriter(output / "ground_truth.jsonl", config, ARGS.seed, navmesh)
    steps = int(ARGS.steps if ARGS.steps is not None else episode["steps"])
    initial = {cfg["id"]: vector(person.state.position) for cfg, person, _ in people}
    try:
        for step in range(steps + 1):
            if not simulation_app.is_running():
                raise RuntimeError("SimulationApp stopped before the episode completed")
            if step > 0:
                world.step(render=True)
                for _, _, controller in people:
                    controller.enforce_deterministic_pose()
            writer.write(step, float(episode["physics_dt_s"]), people, asset_url)
    finally:
        writer.close()
        timeline.stop()
    movement = {
        cfg["id"]: float(np.linalg.norm(vector(person.state.position)[:2] - initial[cfg["id"]][:2]))
        for cfg, person, _ in people
    }
    rounded_digest = hashlib.sha256()
    for line in (output / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        canonical = [
            [p["id"], *[round(v, 4) for v in p["position_xyz_m"]]]
            for p in record["pedestrians"]
        ]
        rounded_digest.update(json.dumps(canonical, separators=(",", ":")).encode("utf-8"))
    return {
        "status": "pass", "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": "isaac-sim-pegasus", "scenario": config["scenario"]["name"],
        "split": ARGS.split, "episode_index": ARGS.episode_index, "seed": ARGS.seed,
        "official_isaac_asset": asset_url, "steps": steps, "pedestrian_count": len(people),
        "movement_m": movement, "spawn_snap_distance_m": spawn_snaps,
        "max_navmesh_distance_m": writer.max_navmesh_distance_m,
        "zone_count_ranges": writer.zone_ranges,
        "resolved_routes": {
            "artifact": "resolved_routes.json",
            "source": "generated_from_baked_navmesh" if cached_routes is None else "provided_episode_cache",
        },
        "trajectory_digest_4dp": rounded_digest.hexdigest(),
        "controller": {
            "base_api": "pegasus.simulator.logic.people.person_controller.PersonController",
            "static_obstacle_avoidance": "baked Isaac Sim NavMesh shortest paths",
            "social_interaction": "deterministic priority yielding (not SFM or ORCA)",
        },
        "elapsed_s": time.perf_counter() - started,
    }


if __name__ == "__main__":
    result: dict[str, object]
    exit_code = 0
    try:
        result = run()
    except Exception as error:
        exit_code = 1
        result = {
            "status": "fail", "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        ARGS.output.mkdir(parents=True, exist_ok=True)
        (ARGS.output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        simulation_app.close()
    raise SystemExit(exit_code)
