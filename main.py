import src.XML_constellation.constellation_entity.user as USER
import src.constellation_generation.by_XML.constellation_configuration as constellation_configuration
import src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager as connectivity_mode_plugin_manager
import src.XML_constellation.constellation_evaluation.exists_ISL.delay as DELAY
# from entity import *
# from utils import *
import math
from typing import Dict, List, Tuple
from equal_area_partition import EARTH_RADIUS_KM, EqualAreaGrid, EqualAreaCell
import src.constellation_generation.by_XML.constellation_configuration as constellation_configuration
import random
import numpy as np
import matplotlib.pyplot as plt
from pop_user_allocator import build_users_by_population, TOTAL_USERS, POP_TIF_PATH, PopulationAllocation
from utils import plot_andSave, latilong_to_descartes, judgePointToSatellite, calculate_distance
from delay import delay
from select_satellite import select_nearest, select_weighted
from entity import T2TuserTraffic, T2CuserFeedback, U2Svisible, satellite_beam, U2Svisible, gatway, central_node
import scipy.constants as C
import argparse

# 仿真参数：总时长 2 小时、时隙 60 秒
SIMULATION_DURATION = 2 * 60 * 60
TIME_SLOT = 60
# 地面中心节点的自定义经纬度
CENTER_NODE_LOCATION: Tuple[float, float] = (104.0, 35.0)

def init_user(
    lat_range=(-60, 60),
    lon_range=(-90, 90),
    delta_a=1000,
    delta_b=1000,
    distribution="density",
    total_users=TOTAL_USERS,
    tif_path=POP_TIF_PATH,
):
    
    delta_a = float(delta_a)
    delta_b = float(delta_b)

    allocation = build_users_by_population(
        USER_module=USER,
        total_users=total_users,
        tif_path=tif_path,
        lat_range=lat_range,
        lon_range=lon_range,
        delta_a=delta_a,
        delta_b=delta_b,
        distribution=distribution,
    )
    return allocation

def init_gatway(capacity):
    gatways = []
    gateway_locations = [
        ("库尔勒", 86.1779, 41.7259),
        ("佳木斯", 130.3189, 46.7993),
        ("雄安", 115.9929, 38.9968),
        ("铜川", 109.0889, 35.1903),
        ("三亚", 109.5119, 18.2528),
    ]
    for name, lon, lat in gateway_locations:
        gatways.append(gatway(lon, lat, capacity, gateway_name=name))
    return gatways

def init_central_node(capacity, gatways):
    """构建与信关站星型互联的地面中心节点。"""

    lon, lat = CENTER_NODE_LOCATION
    center = central_node(lon, lat, capacity, connected_gateways=gatways,node_name="中心节点")
    return center

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
    """Construct T2T connections by pairing approximately antipodal cells."""

    grid = allocation.grid
    visited: set[int] = set()
    connections: List[T2TuserTraffic] = []
    cells = list(grid.cells)

    for cell in cells:
        if cell.cell_id in visited:
            continue

        desired_lat = -cell.latitude
        desired_lon = _normalise_longitude(cell.longitude + 180.0)

        best_cell = None
        best_score = math.inf
        for candidate in cells:
            if candidate.cell_id == cell.cell_id or candidate.cell_id in visited:
                continue
            score = abs(candidate.latitude - desired_lat) + _angular_distance_deg(candidate.longitude, desired_lon)
            if score < best_score:
                best_score = score
                best_cell = candidate

        if best_cell is None:
            continue

        visited.add(cell.cell_id)
        visited.add(best_cell.cell_id)

        source_users = allocation.cell_user_map.get(cell.cell_id, [])
        target_users = allocation.cell_user_map.get(best_cell.cell_id, [])

        if allocation.distribution == "uniform":
            if not source_users or not target_users:
                continue
            pair_count = min(len(source_users), len(target_users))
            source_pool = list(source_users)
            target_pool = list(target_users)
        else:
            source_pool = list(source_users) if source_users else [_create_virtual_user(cell)]
            target_pool = list(target_users) if target_users else [_create_virtual_user(best_cell)]
            if len(source_pool) == 0 and len(target_pool) == 0:
                continue
            pair_count = max(len(source_pool), len(target_pool))

        for idx in range(pair_count):
            user_a = source_pool[idx % len(source_pool)]
            user_b = target_pool[idx % len(target_pool)]
            connections.append(
                T2TuserTraffic(user_a, user_b, need_bandwidth, max_bandwidth, min_bandwidth)
            )

    return connections
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

def init_satellite(sh, capacity, beam_num, beam_user_num):
    satellites = []
    for orbit in sh.orbits:
        for satellite in orbit.satellites:
            satellite.beam_num = beam_num
            satellite.beams = []
            for i in range(beam_num):
                satellite.beams.append(satellite_beam(beam_user_num, capacity))
            satellites.append(satellite)
    return satellites

def calculate_central_node_load(center: central_node):
    """记录中心节点的瞬时负载。"""

    if center.capacity > 0:
        load = center.bandwidth / center.capacity
    else:
        load = 0
    center.load.append(load)

def calculate_satellite_load(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            load = beam.bandwidth / beam.capacity
            beam.load.append(load)

def calculate_gatway_load(gatways):
    for gatway in gatways:
        load = gatway.bandwidth / gatway.capacity
        gatway.load.append(load)
            
def reset_satellite(satellites):
    for satellite in satellites:
        for beam in satellite.beams:
            beam.bandwidth = 0
            beam.connected_user = 0

def reset_gatway(gatways):
    for gatway in gatways:
        gatway.bandwidth = 0

def reset_central_node(center: central_node):
    """重置中心节点的带宽占用。"""

    center.bandwidth = 0

def init_U2Svisible(users, satellites, t, min_bandwidth, max_bandwidth, min_visible_angle):
    U2Svisible_matrix = []
    for user in users:
        U2Svisible_row = []
        for satellite in satellites:
            U2Svisible_item = U2Svisible(user, satellite, t, min_bandwidth, max_bandwidth, min_visible_angle)
            U2Svisible_row.append(U2Svisible_item)
        U2Svisible_matrix.append(U2Svisible_row)
    return U2Svisible_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_bandwidth', type=int, default=10)
    parser.add_argument('--max_bandwidth', type=int, default=30)
    parser.add_argument('--min_visible_angle', type=int, default=10)
    parser.add_argument('--need_bandwidth', type=int, default=10)
    parser.add_argument('--connect_min_bandwidth', type=int, default=15)
    parser.add_argument('--connect_max_bandwidth', type=int, default=20)
    parser.add_argument('--beam_num', type=int, default=7)
    parser.add_argument('--satellite_capacity', type=int, default=100)
    parser.add_argument('--beam_user_num', type=int, default=8)
    parser.add_argument('--gatway_capacity', type=int, default=1000)
    parser.add_argument('--connect_num', type=int, default=10)
    parser.add_argument(
        '--population_distribution',
        type=str,
        choices=['density', 'uniform'],
        default='density',
        help='人口分布模式：density 表示真实人口权重，uniform 表示全球均匀分布。'
    )
    parser.add_argument(
        '--total_users',
        type=int,
        default=TOTAL_USERS,
        help='总体用户规模，density 模式下生效。'
    )
    args = parser.parse_args()
    run_t2t(args)
    print("t2t done")
    # run_t2c(args)
    # print("t2c done")

def run_t2t(args):
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
    connect_num = args.connect_num
    allocation = init_user(
        distribution=args.population_distribution,
        total_users=args.total_users,
    )
    users = allocation.users
    user_num = len(users)

    connect_tuple = generate_opposite_t2t_connections(
        allocation,
        need_bandwidth,
        connect_max_bandwidth,
        connect_min_bandwidth,
    )
    actual_connect_num = len(connect_tuple)

    constellation_name = 'China'
    dT = TIME_SLOT
    constellation = constellation_configuration.constellation_configuration(dT=dT, constellation_name=constellation_name)
    #108星座
    sh = constellation.shells[0]
    #432星座
    #sh = constellation.shells[1]
    shell_name = sh.shell_name
    satellites = init_satellite(sh, satellite_capacity, beam_num, beam_user_num)

    # U2Svisible_matrix = init_U2Svisible(users, satellites, min_bandwidth, max_bandwidth, min_visible_angle)

    connectionModePluginManager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
    connectionModePluginManager.set_connection_mode(plugin_name="positive_Grid")
    connectionModePluginManager.execute_connection_policy(constellation=constellation , dT=dT)
    
    delay_list = []
    for t in range(1, (int)(sh.orbit_cycle / dT) + 2, 1):
    # 在 2 小时内按 60 秒时隙推进仿真
    # simulation_steps = SIMULATION_DURATION // TIME_SLOT
    # for t in range(1, int(simulation_steps) + 1):
        connect_delay = []
        for connect in connect_tuple:
            user1, user2 = connect.source, connect.target
            connect_users = [user1, user2]
            U2Svisible_matrix = init_U2Svisible(connect_users, satellites, t, min_bandwidth, max_bandwidth, min_visible_angle)
            # source_satellite = select_nearest(user1, satellites, t)
            # target_satellite = select_nearest(user2, satellites, t)
            # user1_U2Svisible_row = U2Svisible_matrix[users.index(user1)]
            # user2_U2Svisible_row = U2Svisible_matrix[users.index(user2)]
            # U2Svisible_matrix = [user1_U2Svisible_row, user2_U2Svisible_row]
            #source_satellite, target_satellite = select_nearest(connect, U2Svisible_matrix, satellites, t, 't2t')
            source_satellite, target_satellite = select_weighted(connect, U2Svisible_matrix, satellites, t, 't2t')
            if source_satellite != None and target_satellite != None:
                source_sat_x, source_sat_y, source_sat_z = latilong_to_descartes(source_satellite, 'satellite', t)
                target_sat_x, target_sat_y, target_sat_z = latilong_to_descartes(target_satellite, 'satellite', t)
                user1_x, user1_y, user1_z = latilong_to_descartes(user1, 'user', t)
                user2_x, user2_y, user2_z = latilong_to_descartes(user2, 'user', t)
                if judgePointToSatellite(source_sat_x, source_sat_y, source_sat_z, user1_x, user1_y, user1_z, min_visible_angle) and judgePointToSatellite(target_sat_x, target_sat_y, target_sat_z, user2_x, user2_y, user2_z, min_visible_angle):
                    u2u_delay = delay(constellation_name, source_satellite, target_satellite, shell_name, t) + 2*calculate_distance(user1, source_satellite, t)/(C.c/1000) + 2*calculate_distance(user2, target_satellite, t)/(C.c/1000)
                    connect_delay.append(u2u_delay)

        calculate_satellite_load(satellites)
        reset_satellite(satellites)
        if connect_delay:
            delay_list.append(connect_delay)

    # 若存在延迟结果，可在此处输出首个样本以便快速检查。
    if delay_list:
        print(delay_list[0])

    
    all_delay = []
    for connect_delay in delay_list:
        all_delay.extend(connect_delay)

    all_beam_load = []
    for satellite in satellites:
        for beam in satellite.beams:
            all_beam_load.extend(beam.load)


    max_bandwidth_ratio_list = []
    min_bandwidth_ratio_list = []
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
    p99_delay = float(np.percentile(all_delay, 99)) if all_delay else 0.0
    p95_beam_load  = float(np.percentile(all_beam_load, 95)) if all_beam_load else 0.0
    p99_beam_load  = float(np.percentile(all_beam_load, 99)) if all_beam_load else 0.0
    p95_max_bandwidth_ratio = float(np.percentile(max_bandwidth_ratio_list, 95)) if max_bandwidth_ratio_list else 0.0
    p95_min_bandwidth_ratio  = float(np.percentile(min_bandwidth_ratio_list, 95)) if min_bandwidth_ratio_list else 0.0
    print(f"t2t p95delay = {p95_delay:.3f}, p99delay = {p99_delay:.3f}, p95load = {p95_beam_load:.3f}, p99load = {p99_beam_load:.3f}, max_bandwidth_ratio={p95_max_bandwidth_ratio:.3f}, min_bandwidth_ratio={p95_min_bandwidth_ratio:.3f}")
    plot_andSave(max_bandwidth_ratio_list, './t2t_max_bandwidth_ratio_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(min_bandwidth_ratio_list, './t2t_min_bandwidth_ratio_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(all_delay, './t2t_delay_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(all_beam_load, './t2t_beam_load_cdf_connect_{}.png'.format(actual_connect_num))

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
    connect_num = args.connect_num
    allocation = init_user(
        distribution=args.population_distribution,
        total_users=args.total_users,
    )
    users = allocation.users
    user_num = len(users)
    gatways = init_gatway(gatway_capacity)
    gatway_num = len(gatways)
    center = init_central_node(gatway_capacity, gatways)
    connect_tuple = generate_centralised_t2c_connections(
        allocation,
        center,
        need_bandwidth,
        connect_max_bandwidth,
        connect_min_bandwidth,
    )
    actual_connect_num = len(connect_tuple)

    constellation_name = 'China'
    dT = TIME_SLOT
    constellation = constellation_configuration.constellation_configuration(dT=dT, constellation_name=constellation_name)
    sh = constellation.shells[0]
    shell_name = sh.shell_name
    satellites = init_satellite(sh, satellite_capacity, beam_num, beam_user_num)

    # U2Svisible_matrix = init_U2Svisible(users, satellites, min_bandwidth, max_bandwidth, min_visible_angle)

    connectionModePluginManager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
    connectionModePluginManager.set_connection_mode(plugin_name="positive_Grid")
    connectionModePluginManager.execute_connection_policy(constellation=constellation , dT=dT)
    
    delay_list = []
    # 在 2 小时内按 60 秒时隙推进仿真
    # simulation_steps = SIMULATION_DURATION // TIME_SLOT
    # for t in range(1, int(simulation_steps) + 1):
    for t in range(1, (int)(sh.orbit_cycle / dT) + 2, 1):
        connect_delay = []
        for connect in connect_tuple:
            user1, user2 = connect.source, connect.target
            connect_users = [user1, user2]
            U2Svisible_matrix = init_U2Svisible(connect_users, satellites, t, min_bandwidth, max_bandwidth, min_visible_angle)
            # source_satellite = select_nearest(user1, satellites, t)
            # target_satellite = select_nearest(user2, satellites, t)
            # user1_U2Svisible_row = U2Svisible_matrix[users.index(user1)]
            # user2_U2Svisible_row = U2Svisible_matrix[users.index(user2)]
            # U2Svisible_matrix = [user1_U2Svisible_row, user2_U2Svisible_row]
            #source_satellite, target_satellite = select_nearest(connect, U2Svisible_matrix, satellites, t, 't2c')
            source_satellite, target_satellite = select_weighted(connect, U2Svisible_matrix, satellites, t, 't2c')
            if source_satellite != None and target_satellite != None:
                source_sat_x, source_sat_y, source_sat_z = latilong_to_descartes(source_satellite, 'satellite', t)
                target_sat_x, target_sat_y, target_sat_z = latilong_to_descartes(target_satellite, 'satellite', t)
                user1_x, user1_y, user1_z = latilong_to_descartes(user1, 'user', t)
                user2_x, user2_y, user2_z = latilong_to_descartes(user2, 'user', t)
                if judgePointToSatellite(source_sat_x, source_sat_y, source_sat_z, user1_x, user1_y, user1_z, min_visible_angle) and judgePointToSatellite(target_sat_x, target_sat_y, target_sat_z, user2_x, user2_y, user2_z, min_visible_angle):
                    u2u_delay = delay(constellation_name, source_satellite, target_satellite, shell_name, t) + 2*calculate_distance(user1, source_satellite, t)/(C.c/1000) + 2*calculate_distance(user2, target_satellite, t)/(C.c/1000)
                    connect_delay.append(u2u_delay)

        calculate_satellite_load(satellites)
        calculate_gatway_load(gatways)
        calculate_central_node_load(center)
        reset_satellite(satellites)
        reset_gatway(gatways)
        reset_central_node(center)
        if connect_delay:
            delay_list.append(connect_delay)

    # cdf_list = [item[0] for item in delay_list]
    # print(delay_list[0])

    
    all_delay = []
    for connect_delay in delay_list:
        all_delay.extend(connect_delay)

    all_beam_load = []
    for satellite in satellites:
        for beam in satellite.beams:
            all_beam_load.extend(beam.load)
    
    all_gatway_load = []
    for gatway in gatways:
        all_gatway_load.extend(gatway.load)

    all_center_load = list(center.load)

    max_bandwidth_ratio_list = []
    min_bandwidth_ratio_list = []
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
    p95_beam_load  = float(np.percentile(all_beam_load, 95)) if all_beam_load else 0.0
    p95_gateway_load = float(np.percentile(all_gatway_load, 95)) if all_gatway_load else 0.0
    p95_center_load = float(np.percentile(all_center_load, 95)) if all_center_load else 0.0
    p95_max_bandwidth_ratio = float(np.percentile(max_bandwidth_ratio_list, 95)) if max_bandwidth_ratio_list else 0.0
    p95_min_bandwidth_ratio  = float(np.percentile(min_bandwidth_ratio_list, 95)) if min_bandwidth_ratio_list else 0.0
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
    plot_andSave(max_bandwidth_ratio_list, './t2c_max_bandwidth_ratio_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(min_bandwidth_ratio_list, './t2c_min_bandwidth_ratio_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(all_delay, './t2c_delay_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(all_beam_load, './t2c_beam_load_cdf_connect_{}.png'.format(actual_connect_num))
    plot_andSave(all_gatway_load, './t2c_gatway_load_cdf_connect_{}.png'.format(actual_connect_num))
    if all_center_load:
        plot_andSave(all_center_load, './t2c_center_load_cdf_connect_{}.png'.format(actual_connect_num))
if __name__ == '__main__':
    main()
