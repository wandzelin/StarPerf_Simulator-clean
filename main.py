"""Entry point for StarPerf user-to-user (T2T) simulations."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import src.XML_constellation.constellation_entity.user as USER
import src.constellation_generation.by_XML.constellation_configuration as constellation_configuration
import src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager as connectivity_mode_plugin_manager
from delay import delay
from entity import (
    T2TuserTraffic,
    U2Svisible,
    central_node,
    gatway,
    satellite_beam,
)
from equal_area_partition import EqualAreaCell, EqualAreaGrid
from pop_user_allocator import POP_TIF_PATH, TOTAL_USERS, PopulationAllocation, build_users_by_population
from select_satellite import select_weighted, set_selection_mode
from utils import calculate_distance, latilong_to_descartes, plot_andSave, judgePointToSatellite
import scipy.constants as C

# 仿真参数：总时长 2 小时、时隙 60 秒
SIMULATION_DURATION = 2 * 60 * 60
TIME_SLOT = 60
DEFAULT_CENTER_NODE_LOCATION: Tuple[float, float] = (104.0, 35.0)


@dataclass
class SimulationConfig:
    min_bandwidth: int = 10
    max_bandwidth: int = 30
    min_visible_angle: int = 10
    need_bandwidth: int = 10
    connect_min_bandwidth: int = 15
    connect_max_bandwidth: int = 20
    beam_num: int = 7
    satellite_capacity: int = 100
    beam_user_num: int = 8
    gatway_capacity: int = 1000
    connect_num: int = 10
    population_distribution: str = "uniform"  # "density" or "uniform"
    selection_mode: str = "nearest"  # "nearest" or "weighted"
    shell: str = "shell2"  # "shell1" or "shell2"
    density_total_users: int = TOTAL_USERS
    total_users: int = TOTAL_USERS
    constellation_name: str = "China"
    center_node_location: Tuple[float, float] = DEFAULT_CENTER_NODE_LOCATION

    def validate(self) -> "SimulationConfig":
        """Ensure numeric values are positive where required."""

        ints_to_check = {
            "min_bandwidth": self.min_bandwidth,
            "max_bandwidth": self.max_bandwidth,
            "need_bandwidth": self.need_bandwidth,
            "connect_min_bandwidth": self.connect_min_bandwidth,
            "connect_max_bandwidth": self.connect_max_bandwidth,
            "beam_num": self.beam_num,
            "satellite_capacity": self.satellite_capacity,
            "beam_user_num": self.beam_user_num,
            "gatway_capacity": self.gatway_capacity,
        }
        for name, value in ints_to_check.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.population_distribution not in {"density", "uniform"}:
            raise ValueError(
                "population_distribution must be either 'density' or 'uniform'"
            )
        if self.selection_mode not in {"nearest", "weighted"}:
            raise ValueError("selection_mode must be either 'nearest' or 'weighted'")
        if self.shell not in {"shell1", "shell2"}:
            raise ValueError("shell must be either 'shell1' or 'shell2'")
        if self.density_total_users < 0:
            raise ValueError("density_total_users must be non-negative")
        if self.total_users < 0:
            raise ValueError("total_users must be non-negative")
        return self

def init_user(

    
    lat_range: Tuple[float, float] = (-60, 60),
    lon_range: Tuple[float, float] = (-90, 90),
    delta_a: float = 1000,
    delta_b: float = 1000,
    distribution: str = "density",
    total_users: int = TOTAL_USERS,
    tif_path: str = POP_TIF_PATH,
) -> PopulationAllocation:
    """Create user objects by delegating to the population allocator."""

    return build_users_by_population(

        USER_module=USER,
        total_users=total_users,
        tif_path=tif_path,
        lat_range=lat_range,
        lon_range=lon_range,
        delta_a=float(delta_a),
        delta_b=float(delta_b),
        distribution=distribution,
    )

def init_gatway(capacity: int) -> List[gatway]:
    """初始化固定的地面信关站列表。"""

    gateway_locations = [
        ("库尔勒", 86.1779, 41.7259),
        ("佳木斯", 130.3189, 46.7993),
        ("雄安", 115.9929, 38.9968),
        ("铜川", 109.0889, 35.1903),
        ("三亚", 109.5119, 18.2528),
    ]
    return [
        gatway(lon, lat, capacity, gateway_name=name)
        for name, lon, lat in gateway_locations
    ]
def init_central_node(
    capacity: int,
    gatways: Sequence[gatway],
    location: Tuple[float, float] | None = None,
) -> central_node:
    """构建与信关站星型互联的地面中心节点。"""

    if location is None:
        location = DEFAULT_CENTER_NODE_LOCATION
    lon, lat = location
    return central_node(
        lon,
        lat,
        capacity,
        connected_gateways=list(gatways),
        node_name="中心节点",
    )

def _normalise_longitude(lon: float) -> float:
    """将经度归一化到 [-180, 180) 区间，便于建立对置小区索引。"""

    normalised = (lon + 180.0) % 360.0
    if normalised < 0:
        normalised += 360.0
    return normalised - 180.0

def _angular_distance_deg(lon_a: float, lon_b: float) -> float:
    """经度差大于180则折算成另一侧的差值。"""
    diff = abs(lon_a - lon_b)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff

#grid=小区类 grid.cells=小区列表 cell=小区
def _build_cell_lookup(grid: EqualAreaGrid) -> Dict[Tuple[float, float], int]:
    """以小区中心点坐标为键索引小区id，用于快速根据对置小区中心点坐标查找对置小区id。"""

    lookup: Dict[Tuple[float, float], int] = {}
    for cell in grid.cells:
        key = (round(cell.latitude, 6), round(_normalise_longitude(cell.longitude), 6))
        lookup[key] = cell.cell_id
    return lookup

def _create_virtual_user(cell: EqualAreaCell):
    """为缺失用户的小区生成临时用户，用于保持连接逻辑完整。"""

    user_obj = USER.user(cell.latitude, cell.longitude)
    setattr(user_obj, "cell_id", cell.cell_id)
    setattr(user_obj, "cell_row", cell.i)
    setattr(user_obj, "cell_col", cell.j)
    setattr(user_obj, "cell_latitude", cell.latitude)
    setattr(user_obj, "cell_longitude", cell.longitude)
    return user_obj

def generate_opposite_t2t_connections(
    allocation: PopulationAllocation,
    need_bandwidth: int,
    max_bandwidth: int,
    min_bandwidth: int,
) -> List[T2TuserTraffic]:
    """通过对置小区配对构造 T2T 连接。"""

    connections: List[T2TuserTraffic] = []
    visited: set[int] = set()
    cells = list(allocation.grid.cells)

    for cell in cells:
        if cell.cell_id in visited:
            continue

        partner = _find_opposite_cell(cell, cells, visited)
        if partner is None:
            continue

        # 标记已配对的小区，避免重复建立连接
        visited.add(cell.cell_id)
        visited.add(partner.cell_id)

        source_pool = allocation.cell_user_map.get(cell.cell_id, [])
        target_pool = allocation.cell_user_map.get(partner.cell_id, [])

        for user_a, user_b in _iterate_user_pairs(
            source_pool,
            target_pool,
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
    """寻找当前小区的最佳对置小区。"""

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


def _iterate_user_pairs(
    source_users: Sequence,
    target_users: Sequence,
    source_cell: EqualAreaCell,
    target_cell: EqualAreaCell,
    distribution: str,
):
    """根据人口分布策略生成用户对。"""

    if distribution == "uniform":
        if not source_users or not target_users:
            return
        pair_count = min(len(source_users), len(target_users))
        for idx in range(pair_count):
            yield source_users[idx], target_users[idx]
        return

    source_pool = list(source_users) if source_users else [_create_virtual_user(source_cell)]
    target_pool = list(target_users) if target_users else [_create_virtual_user(target_cell)]
    if not source_pool and not target_pool:
        return

    pair_count = max(len(source_pool), len(target_pool))
    for idx in range(pair_count):
        yield source_pool[idx % len(source_pool)], target_pool[idx % len(target_pool)]


def generate_centralised_t2c_connections(
    allocation: PopulationAllocation,
    center: central_node,
    need_bandwidth: int,
    max_bandwidth: int,
    min_bandwidth: int,
) -> List[T2TuserTraffic]:
    """为每个用户构造仅指向中心节点的 T2C 连接。"""

    connections: List[T2TuserTraffic] = []
    for user in allocation.users:
        connections.append(T2TuserTraffic(user, center, need_bandwidth, max_bandwidth, min_bandwidth))
    return connections

def init_satellite(shell, capacity: int, beam_num: int, beam_user_num: int):
    satellites = []
    for orbit in shell.orbits:
        for sat in orbit.satellites:
            sat.beam_num = beam_num
            sat.beams = [satellite_beam(beam_user_num, capacity) for _ in range(beam_num)]
            satellites.append(sat)
    return satellites

def calculate_central_node_load(center: central_node):
    """记录中心节点的瞬时负载。"""

    load = center.bandwidth / center.capacity if center.capacity else 0
    center.load.append(load)

def calculate_satellite_load(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            beam.load.append(beam.bandwidth / beam.capacity)

def calculate_gatway_load(gatways):
    for gate in gatways:
        gate.load.append(gate.bandwidth / gate.capacity)
            
def reset_satellite(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            beam.bandwidth = 0
            beam.connected_user = 0

def reset_gatway(gatways):
    for gate in gatways:
        gate.bandwidth = 0

def reset_central_node(center: central_node):
    """重置中心节点的带宽占用。"""

    center.bandwidth = 0

def init_U2Svisible(users, satellites, t, min_bandwidth, max_bandwidth, min_visible_angle):
    matrix = []
    for user in users:
        row = [
            U2Svisible(user, satellite, t, min_bandwidth, max_bandwidth, min_visible_angle)
            for satellite in satellites
        ]
        matrix.append(row)
    return matrix


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser()
    defaults = SimulationConfig().validate()
    parser.set_defaults(**vars(defaults))
    parser.add_argument("--min_bandwidth", type=int, help="链路最小带宽")
    parser.add_argument("--max_bandwidth", type=int, help="链路最大带宽")
    parser.add_argument("--min_visible_angle", type=int, help="卫星可见最小仰角")
    parser.add_argument("--need_bandwidth", type=int, help="业务所需带宽")
    parser.add_argument("--connect_min_bandwidth", type=int, help="连接最小带宽需求")
    parser.add_argument("--connect_max_bandwidth", type=int, help="连接最大带宽需求")
    parser.add_argument("--beam_num", type=int, help="卫星波束数量")
    parser.add_argument("--satellite_capacity", type=int, help="卫星总容量")
    parser.add_argument("--beam_user_num", type=int, help="单波束可服务用户数")
    parser.add_argument("--gatway_capacity", type=int, help="地面网关容量")
    parser.add_argument("--connect_num", type=int, help="连接数量")
    parser.add_argument(
        "--population_distribution",
        type=str,
        choices=["density", "uniform"],
        help="人口分布模式（density/uniform）",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["nearest", "weighted"],
        help="卫星选择策略（nearest/weighted）",
    )
    parser.add_argument(
        "--density_total_users",
        type=int,
        help="人口密度模式下的总用户数",
    )
    parser.add_argument(
        "--shell",
        type=str,
        choices=["shell1", "shell2"],
        help="星座壳层选择：shell1=108 星，shell2=432 星",
    )
    parser.add_argument(
        "--total_users",
        type=int,
        help="均匀模式下自定义总用户数",
    )
    parser.add_argument(
        "--constellation_name",
        type=str,
        help="运行仿真的星座名称",
    )
    args = parser.parse_args()
    config = SimulationConfig(**vars(args)).validate()
    return config





def main():
    config = parse_args()
    set_selection_mode(config.selection_mode)
    if config.population_distribution == "density":
        config.total_users = config.density_total_users
    run_t2t(config)

    print("t2t done")
    run_t2c(config)
    print("t2c done")



def _select_constellation_shell(constellation, shell_key: str):
    """Resolve shell identifier from CLI config to actual constellation shell."""

    mapping = {"shell1": 0, "shell2": 1}
    try:
        index = mapping[shell_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported shell option: {shell_key}") from exc
    try:
        return constellation.shells[index]
    except IndexError as exc:
        raise ValueError(f"Shell index {index} unavailable in constellation configuration") from exc


def run_t2t(config: SimulationConfig):
    """运行端到端 T2T 仿真流程。"""

    allocation = init_user(
        distribution=config.population_distribution,
        total_users=config.total_users,
    )

    # 依据对置小区生成端到端用户连接
    connections = generate_opposite_t2t_connections(
        allocation,
        config.need_bandwidth,
        config.connect_max_bandwidth,
        config.connect_min_bandwidth,
    )

    constellation = constellation_configuration.constellation_configuration(
        dT=TIME_SLOT,
        constellation_name=config.constellation_name,
    )
    shell = _select_constellation_shell(constellation, config.shell)
    satellites = init_satellite(
        shell,
        config.satellite_capacity,
        config.beam_num,
        config.beam_user_num,
    )

    # 设置并执行连接策略，确保卫星网络拓扑已更新
    connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
    connection_manager.set_connection_mode(plugin_name="positive_Grid")
    connection_manager.execute_connection_policy(
        constellation=constellation,
        dT=TIME_SLOT,
    )

    delay_list: List[List[float]] = []
    shell_cycle_steps = int(shell.orbit_cycle / TIME_SLOT) + 2

    # 按时隙遍历，逐个连接计算链路时延
    for t in range(1, shell_cycle_steps):
        slot_delays: List[float] = []
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

        calculate_satellite_load(satellites)
        reset_satellite(satellites)
        if slot_delays:
            delay_list.append(slot_delays)

    if delay_list:
        print(delay_list[0])

    all_delay = [value for slot in delay_list for value in slot]
    all_beam_load = [
        load
        for sat in satellites
        for beam in sat.beams
        for load in beam.load
    ]

    max_bandwidth_ratio_list, min_bandwidth_ratio_list = _summarise_bandwidth_ratios(
        connections
    )

    p95_delay = float(np.percentile(all_delay, 95)) if all_delay else 0.0
    p99_delay = float(np.percentile(all_delay, 99)) if all_delay else 0.0
    p95_beam_load = float(np.percentile(all_beam_load, 95)) if all_beam_load else 0.0
    p99_beam_load = float(np.percentile(all_beam_load, 99)) if all_beam_load else 0.0
    p95_max_bandwidth_ratio = (
        float(np.percentile(max_bandwidth_ratio_list, 95))
        if max_bandwidth_ratio_list
        else 0.0
    )
    p95_min_bandwidth_ratio = (
        float(np.percentile(min_bandwidth_ratio_list, 95))
        if min_bandwidth_ratio_list
        else 0.0
    )

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
        max_bandwidth_ratio_list,
        f"./t2t_max_bandwidth_ratio_cdf_connect_{len(connections)}.png",
    )


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
    # 构建用户与卫星的可见性矩阵，用于后续卫星选择
    visibility = init_U2Svisible(
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

    source_sat_x, source_sat_y, source_sat_z = latilong_to_descartes(
        source_satellite, "satellite", t
    )
    target_sat_x, target_sat_y, target_sat_z = latilong_to_descartes(
        target_satellite, "satellite", t
    )
    user1_x, user1_y, user1_z = latilong_to_descartes(users[0], "user", t)
    user2_x, user2_y, user2_z = latilong_to_descartes(users[1], "user", t)

    # 判断用户与选中卫星之间的可见性是否满足仰角要求
    if not (
        judgePointToSatellite(
            source_sat_x,
            source_sat_y,
            source_sat_z,
            user1_x,
            user1_y,
            user1_z,
            min_visible_angle,
        )
        and judgePointToSatellite(
            target_sat_x,
            target_sat_y,
            target_sat_z,
            user2_x,
            user2_y,
            user2_z,
            min_visible_angle,
        )
    ):
        return None

    return (
        delay(constellation_name, source_satellite, target_satellite, shell_name, t)
        + 2 * calculate_distance(users[0], source_satellite, t) / (C.c / 1000)
        + 2 * calculate_distance(users[1], target_satellite, t) / (C.c / 1000)
    )


def _summarise_bandwidth_ratios(connections: Sequence[T2TuserTraffic]):
    """统计连接达到最小/最大带宽需求的比例。"""

    max_ratios: List[float] = []
    min_ratios: List[float] = []

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

def run_t2c(args):
    min_bandwidth = args.min_bandwidth
    max_bandwidth = args.max_bandwidth
    min_visible_angle = args.min_visible_angle
    need_bandwidth = args.need_bandwidth
    connect_min_bandwidth = args.connect_min_bandwidth
    connect_max_bandwidth = args.connect_max_bandwidth
    beam_num = args.beam_num
    satellite_capacity = args.satellite_capacity
    beam_user_num = args.beam_user_num
    gatway_capacity = args.gatway_capacity

    allocation = init_user(
        distribution=args.population_distribution,
        total_users=args.total_users,
    )
    gatways = init_gatway(gatway_capacity)
    center = init_central_node(gatway_capacity, gatways)
    connect_tuple = generate_centralised_t2c_connections(
        allocation,
        center,
        need_bandwidth,
        connect_max_bandwidth,
        connect_min_bandwidth,
    )
    actual_connect_num = len(connect_tuple)

    constellation_name = "China"
    dT = TIME_SLOT
    constellation = constellation_configuration.constellation_configuration(
        dT=dT,
        constellation_name=constellation_name,
    )
    sh = _select_constellation_shell(constellation, args.shell)
    shell_name = sh.shell_name
    satellites = init_satellite(sh, satellite_capacity, beam_num, beam_user_num)

    connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
    connection_manager.set_connection_mode(plugin_name="positive_Grid")
    connection_manager.execute_connection_policy(constellation=constellation, dT=dT)

    delay_list: List[List[float]] = []
    for t in range(1, int(sh.orbit_cycle / dT) + 2):
        connect_delay: List[float] = []
        for connect in connect_tuple:
            user1, user2 = connect.source, connect.target
            connect_users = [user1, user2]
            visibility_matrix = init_U2Svisible(
                connect_users,
                satellites,
                t,
                min_bandwidth,
                max_bandwidth,
                min_visible_angle,
            )
            source_satellite, target_satellite = select_weighted(
                connect,
                visibility_matrix,
                satellites,
                t,
                "t2c",
            )
            if source_satellite is not None and target_satellite is not None:
                source_sat_x, source_sat_y, source_sat_z = latilong_to_descartes(
                    source_satellite,
                    "satellite",
                    t,
                )
                target_sat_x, target_sat_y, target_sat_z = latilong_to_descartes(
                    target_satellite,
                    "satellite",
                    t,
                )
                user1_x, user1_y, user1_z = latilong_to_descartes(user1, "user", t)
                user2_x, user2_y, user2_z = latilong_to_descartes(user2, "user", t)
                if (
                    judgePointToSatellite(
                        source_sat_x,
                        source_sat_y,
                        source_sat_z,
                        user1_x,
                        user1_y,
                        user1_z,
                        min_visible_angle,
                    )
                    and judgePointToSatellite(
                        target_sat_x,
                        target_sat_y,
                        target_sat_z,
                        user2_x,
                        user2_y,
                        user2_z,
                        min_visible_angle,
                    )
                ):
                    u2u_delay = (
                        delay(
                            constellation_name,
                            source_satellite,
                            target_satellite,
                            shell_name,
                            t,
                        )
                        + 2
                        * calculate_distance(user1, source_satellite, t)
                        / (C.c / 1000)
                        + 2
                        * calculate_distance(user2, target_satellite, t)
                        / (C.c / 1000)
                    )
                    connect_delay.append(u2u_delay)

        calculate_satellite_load(satellites)
        calculate_gatway_load(gatways)
        calculate_central_node_load(center)
        reset_satellite(satellites)
        reset_gatway(gatways)
        reset_central_node(center)
        if connect_delay:
            delay_list.append(connect_delay)

    all_delay: List[float] = []
    for connect_delay in delay_list:
        all_delay.extend(connect_delay)

    all_beam_load: List[float] = []
    for satellite in satellites:
        for beam in satellite.beams:
            all_beam_load.extend(beam.load)

    all_gatway_load: List[float] = []
    for gatway in gatways:
        all_gatway_load.extend(gatway.load)

    all_center_load = list(center.load)

    max_bandwidth_ratio_list: List[float] = []
    min_bandwidth_ratio_list: List[float] = []
    for connect in connect_tuple:
        max_bandwidth_ratio = 0
        min_bandwidth_ratio = 0
        for allocated_bandwidth in connect.allocated_bandwidth:
            if allocated_bandwidth >= connect.max_bandwidth:
                max_bandwidth_ratio += 1
            if allocated_bandwidth >= connect.min_bandwidth:
                min_bandwidth_ratio += 1
        if connect.allocated_bandwidth:
            length = len(connect.allocated_bandwidth)
            max_bandwidth_ratio_list.append(max_bandwidth_ratio / length)
            min_bandwidth_ratio_list.append(min_bandwidth_ratio / length)

    p95_delay = float(np.percentile(all_delay, 95)) if all_delay else 0.0
    p95_beam_load = float(np.percentile(all_beam_load, 95)) if all_beam_load else 0.0
    p95_gateway_load = float(np.percentile(all_gatway_load, 95)) if all_gatway_load else 0.0
    p95_center_load = float(np.percentile(all_center_load, 95)) if all_center_load else 0.0
    p95_max_bandwidth_ratio = (
        float(np.percentile(max_bandwidth_ratio_list, 95))
        if max_bandwidth_ratio_list
        else 0.0
    )
    p95_min_bandwidth_ratio = (
        float(np.percentile(min_bandwidth_ratio_list, 95))
        if min_bandwidth_ratio_list
        else 0.0
    )
    print(
        "t2c[P95] delay = {:.3f}, beam_load = {:.3f}, gateway_load = {:.3f}, center_load = {:.3f}, "
        "max_bandwidth_ratio = {:.3f}, min_bandwidth_ratio = {:.3f}".format(
            p95_delay,
            p95_beam_load,
            p95_gateway_load,
            p95_center_load,
            p95_max_bandwidth_ratio,
            p95_min_bandwidth_ratio,
        )
    )
    plot_andSave(
        max_bandwidth_ratio_list,
        "./t2c_max_bandwidth_ratio_cdf_connect_{}.png".format(actual_connect_num),
    )
    plot_andSave(
        min_bandwidth_ratio_list,
        "./t2c_min_bandwidth_ratio_cdf_connect_{}.png".format(actual_connect_num),
    )
    plot_andSave(
        all_delay,
        "./t2c_delay_cdf_connect_{}.png".format(actual_connect_num),
    )
    plot_andSave(
        all_beam_load,
        "./t2c_beam_load_cdf_connect_{}.png".format(actual_connect_num),
    )
    plot_andSave(
        all_gatway_load,
        "./t2c_gatway_load_cdf_connect_{}.png".format(actual_connect_num),
    )
    if all_center_load:
        plot_andSave(
            all_center_load,
            "./t2c_center_load_cdf_connect_{}.png".format(actual_connect_num),
        )
if __name__ == '__main__':
    main()
