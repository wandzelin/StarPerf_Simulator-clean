"""Entry point for StarPerf user-to-user (T2T) simulations."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import cycle
from typing import Iterable, Sequence

import numpy as np
import scipy.constants as C

import src.XML_constellation.constellation_entity.user as USER
import src.constellation_generation.by_XML.constellation_configuration as constellation_configuration
import src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager as connectivity_mode_plugin_manager
from delay import delay
from entity import T2TuserTraffic, U2Svisible, satellite_beam
from equal_area_partition import EqualAreaCell
from pop_user_allocator import (
    POP_TIF_PATH,
    TOTAL_USERS,
    PopulationAllocation,
    build_users_by_population,
)
from select_satellite import select_weighted
from utils import calculate_distance, latilong_to_descartes, plot_andSave, judgePointToSatellite

TIME_SLOT = 60


@dataclass
class SimulationConfig:
    """Collect runtime parameters for a T2T simulation."""

    min_bandwidth: int = 10
    max_bandwidth: int = 30
    need_bandwidth: int = 10
    min_visible_angle: int = 10
    beam_num: int = 7
    satellite_capacity: int = 100
    beam_user_num: int = 8
    population_distribution: str = "density"
    total_users: int = TOTAL_USERS
    constellation_name: str = "China"

    def validate(self) -> "SimulationConfig":
        for name in (
            "min_bandwidth",
            "max_bandwidth",
            "need_bandwidth",
            "min_visible_angle",
            "beam_num",
            "satellite_capacity",
            "beam_user_num",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if self.min_bandwidth > self.max_bandwidth:
            raise ValueError("min_bandwidth cannot exceed max_bandwidth")

        if self.population_distribution not in {"density", "uniform"}:
            raise ValueError("population_distribution must be 'density' or 'uniform'")

        if self.total_users < 0:
            raise ValueError("total_users must be non-negative")

        return self


def build_user_allocation(config: SimulationConfig) -> PopulationAllocation:
    """Create ground users on an equal-area grid."""

    return build_users_by_population(
        USER_module=USER,
        total_users=config.total_users,
        tif_path=POP_TIF_PATH,
        lat_range=(-60, 60),
        lon_range=(-90, 90),
        delta_a=1000.0,
        delta_b=1000.0,
        distribution=config.population_distribution,
    )


def generate_opposite_t2t_connections(
    allocation: PopulationAllocation,
    need_bandwidth: int,
    max_bandwidth: int,
    min_bandwidth: int,
) -> list[T2TuserTraffic]:
    """Construct T2T connections by pairing approximately antipodal cells."""

    connections: list[T2TuserTraffic] = []
    visited: set[int] = set()
    cells = list(allocation.grid.cells)

    for cell in cells:
        if cell.cell_id in visited:
            continue

        partner = _find_opposite_cell(cell, cells, visited)
        if partner is None:
            continue

        visited.update({cell.cell_id, partner.cell_id})

        for user_a, user_b in _iter_user_pairs(
            allocation.cell_user_map.get(cell.cell_id, []),
            allocation.cell_user_map.get(partner.cell_id, []),
            cell,
            partner,
            allocation.distribution,
        ):
            connections.append(
                T2TuserTraffic(
                    user_a,
                    user_b,
                    need_bandwidth,
                    max_bandwidth,
                    min_bandwidth,
                )
            )

    return connections


def _find_opposite_cell(
    cell: EqualAreaCell,
    candidates: Sequence[EqualAreaCell],
    visited: Iterable[int],
) -> EqualAreaCell | None:
    desired_lat = -cell.latitude
    desired_lon = _normalise_longitude(cell.longitude + 180.0)

    best_cell: EqualAreaCell | None = None
    best_score = math.inf

    for candidate in candidates:
        if candidate.cell_id == cell.cell_id or candidate.cell_id in visited:
            continue

        score = abs(candidate.latitude - desired_lat) + _angular_distance_deg(
            candidate.longitude, desired_lon
        )
        if score < best_score:
            best_score = score
            best_cell = candidate

    return best_cell


def _normalise_longitude(lon: float) -> float:
    """Normalise longitude into the [-180, 180) range."""

    normalised = (lon + 180.0) % 360.0
    if normalised < 0:
        normalised += 360.0
    return normalised - 180.0


def _angular_distance_deg(lon_a: float, lon_b: float) -> float:
    diff = abs(lon_a - lon_b)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def _iter_user_pairs(
    source_users,
    target_users,
    source_cell: EqualAreaCell,
    target_cell: EqualAreaCell,
    distribution: str,
):
    if distribution == "uniform":
        yield from zip(source_users, target_users)
        return

    source_pool = list(source_users) or [_create_virtual_user(source_cell)]
    target_pool = list(target_users) or [_create_virtual_user(target_cell)]

    source_cycle = cycle(source_pool)
    target_cycle = cycle(target_pool)
    for _ in range(max(len(source_pool), len(target_pool))):
        yield next(source_cycle), next(target_cycle)


def _create_virtual_user(cell: EqualAreaCell):
    """Create a temporary user when a cell is empty."""

    user_obj = USER.user(cell.latitude, cell.longitude)
    setattr(user_obj, "cell_id", cell.cell_id)
    setattr(user_obj, "cell_row", cell.i)
    setattr(user_obj, "cell_col", cell.j)
    setattr(user_obj, "cell_latitude", cell.latitude)
    setattr(user_obj, "cell_longitude", cell.longitude)
    return user_obj


def init_satellite(shell, capacity: int, beam_num: int, beam_user_num: int):
    satellites = []
    for orbit in shell.orbits:
        for sat in orbit.satellites:
            sat.beam_num = beam_num
            sat.beams = [satellite_beam(beam_user_num, capacity) for _ in range(beam_num)]
            satellites.append(sat)
    return satellites


def _record_beam_loads(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            if beam.capacity:
                beam.load.append(beam.bandwidth / beam.capacity)
            else:
                beam.load.append(0.0)


def _reset_satellites(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            beam.bandwidth = 0
            beam.connected_user = 0


def _build_visibility_matrix(
    users,
    satellites,
    t: int,
    min_bandwidth: int,
    max_bandwidth: int,
    min_visible_angle: int,
):
    return [
        [
            U2Svisible(user, satellite, t, min_bandwidth, max_bandwidth, min_visible_angle)
            for satellite in satellites
        ]
        for user in users
    ]


def _evaluate_connection_delay(
    connection: T2TuserTraffic,
    satellites,
    t: int,
    min_bandwidth: int,
    max_bandwidth: int,
    min_visible_angle: int,
    constellation_name: str,
    shell_name: str,
):
    users = [connection.source, connection.target]
    visibility = _build_visibility_matrix(
        users,
        satellites,
        t,
        min_bandwidth,
        max_bandwidth,
        min_visible_angle,
    )
    source_satellite, target_satellite = select_weighted(
        connection, visibility, satellites, t, "t2t"
    )
    if source_satellite is None or target_satellite is None:
        return None

    source_coords = latilong_to_descartes(source_satellite, "satellite", t)
    target_coords = latilong_to_descartes(target_satellite, "satellite", t)
    user1_coords = latilong_to_descartes(users[0], "user", t)
    user2_coords = latilong_to_descartes(users[1], "user", t)

    if not (
        judgePointToSatellite(*source_coords, *user1_coords, min_visible_angle)
        and judgePointToSatellite(*target_coords, *user2_coords, min_visible_angle)
    ):
        return None

    return (
        delay(constellation_name, source_satellite, target_satellite, shell_name, t)
        + 2 * calculate_distance(users[0], source_satellite, t) / (C.c / 1000)
        + 2 * calculate_distance(users[1], target_satellite, t) / (C.c / 1000)
    )


def _summarise_bandwidth_ratios(
    connections: Sequence[T2TuserTraffic],
) -> tuple[list[float], list[float]]:
    max_ratios: list[float] = []
    min_ratios: list[float] = []

    for connection in connections:
        allocations = connection.allocated_bandwidth
        if not allocations:
            continue

        satisfied_max = sum(1 for value in allocations if value >= connection.max_bandwidth)
        satisfied_min = sum(1 for value in allocations if value >= connection.min_bandwidth)
        length = len(allocations)
        max_ratios.append(satisfied_max / length)
        min_ratios.append(satisfied_min / length)

    return max_ratios, min_ratios


def run_t2t(config: SimulationConfig) -> None:
    allocation = build_user_allocation(config)

    connections = generate_opposite_t2t_connections(
        allocation,
        config.need_bandwidth,
        config.max_bandwidth,
        config.min_bandwidth,
    )

    constellation = constellation_configuration.constellation_configuration(
        dT=TIME_SLOT, constellation_name=config.constellation_name
    )
    shell = constellation.shells[0]
    satellites = init_satellite(
        shell,
        config.satellite_capacity,
        config.beam_num,
        config.beam_user_num,
    )

    plugin_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
    plugin_manager.set_connection_mode(plugin_name="positive_Grid")
    plugin_manager.execute_connection_policy(constellation=constellation, dT=TIME_SLOT)

    slot_count = int(math.ceil(shell.orbit_cycle / TIME_SLOT)) + 1

    all_delays: list[float] = []
    for t in range(1, slot_count + 1):
        slot_delays: list[float] = []
        for connection in connections:
            delay_value = _evaluate_connection_delay(
                connection,
                satellites,
                t,
                config.min_bandwidth,
                config.max_bandwidth,
                config.min_visible_angle,
                config.constellation_name,
                shell.shell_name,
            )
            if delay_value is not None:
                slot_delays.append(delay_value)

        if slot_delays:
            all_delays.extend(slot_delays)

        _record_beam_loads(satellites)
        _reset_satellites(satellites)

    beam_load_samples = [
        load
        for satellite in satellites
        for beam in satellite.beams
        for load in beam.load
    ]

    max_ratios, min_ratios = _summarise_bandwidth_ratios(connections)

    p95_delay = float(np.percentile(all_delays, 95)) if all_delays else 0.0
    p99_delay = float(np.percentile(all_delays, 99)) if all_delays else 0.0
    p95_beam_load = float(np.percentile(beam_load_samples, 95)) if beam_load_samples else 0.0
    p99_beam_load = float(np.percentile(beam_load_samples, 99)) if beam_load_samples else 0.0
    p95_max_bandwidth_ratio = float(np.percentile(max_ratios, 95)) if max_ratios else 0.0
    p95_min_bandwidth_ratio = float(np.percentile(min_ratios, 95)) if min_ratios else 0.0

    print(
        "t2t p95delay = {0:.3f}, p99delay = {1:.3f}, p95load = {2:.3f}, "
        "p99load = {3:.3f}, max_bandwidth_ratio={4:.3f}, min_bandwidth_ratio={5:.3f}".format(
            p95_delay,
            p99_delay,
            p95_beam_load,
            p99_beam_load,
            p95_max_bandwidth_ratio,
            p95_min_bandwidth_ratio,
        )
    )

    plot_andSave(
        max_ratios,
        f"./t2t_max_bandwidth_ratio_cdf_connect_{len(connections)}.png",
    )


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description="Run a StarPerf T2T simulation")
    parser.add_argument("--min_bandwidth", type=int, default=10)
    parser.add_argument("--max_bandwidth", type=int, default=30)
    parser.add_argument("--need_bandwidth", type=int, default=10)
    parser.add_argument("--min_visible_angle", type=int, default=10)
    parser.add_argument("--beam_num", type=int, default=7)
    parser.add_argument("--satellite_capacity", type=int, default=100)
    parser.add_argument("--beam_user_num", type=int, default=8)
    parser.add_argument(
        "--population_distribution",
        choices=["density", "uniform"],
        default="density",
        help="density uses population raster weights; uniform splits users evenly.",
    )
    parser.add_argument(
        "--total_users",
        type=int,
        default=TOTAL_USERS,
        help="Total number of simulated users.",
    )
    parser.add_argument(
        "--constellation_name",
        type=str,
        default="China",
        help="Name of the constellation to load from XML/HDF5 files.",
    )
    return SimulationConfig(**vars(parser.parse_args())).validate()


def main() -> None:
    run_t2t(parse_args())


if __name__ == "__main__":
    main()
