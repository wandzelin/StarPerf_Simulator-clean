import math
from utils import *

class T2TuserTraffic:
    def __init__(self, source, target, need_bandwidth, max_bandwidth=None, min_bandwidth=None):
        self.source = source
        self.target = target
        self.need_bandwidth = need_bandwidth
        self.max_bandwidth = max_bandwidth
        self.min_bandwidth = min_bandwidth
        self.allocated_bandwidth = []

class T2CuserFeedback:
    def __init__(self, source, need_bandwidth, max_bandwidth, min_bandwidth):
        self.source = source
        self.target = None
        self.need_bandwidth = need_bandwidth
        self.max_bandwidth = max_bandwidth
        self.min_bandwidth = min_bandwidth


class U2Svisible:
    # Each tuple defines (仰角下限, 仰角上限, MCS等级)
    MCS_RANGES = [
        (20.0, 25.0, 0),
        (25.0, 30.0, 2),
        (30.0, 35.0, 5),
        (35.0, 40.0, 8),
        (40.0, 50.0, 10),
        (50.0, 65.0, 16),
        (65.0, 80.0, 21),
        (80.0, 90.0, 25),
    ]
    # MCS 等级到具体带宽速率（Mbps）的映射
    MCS_BANDWIDTH_MAP = {
        0: 34.5,
        2: 57.0,
        5: 90.0,
        8: 153.0,
        10: 199.5,
        16: 385.5,
        21: 537.0,
        25: 660.0,
    }


    def __init__(self, user, satellite, t, min_bandwidth, max_bandwidth, min_visible_angle) -> None:
        self.user = user
        self.satellite = satellite
        self.t = t
        self.distance = calculate_distance(user, satellite, t)
        self.bandwidth = 0
        self.mcs_level = None

        sat_x, sat_y, sat_z = latilong_to_descartes(satellite, 'satellite', t)
        user_x, user_y, user_z = latilong_to_descartes(user, 'user', t)
        angle = calculate_angle(sat_x, sat_y, sat_z, user_x, user_y, user_z) - 90

        min_required_angle = max(float(min_visible_angle), 20.0)
        if angle < min_required_angle:
            return

        mcs_level = self._resolve_mcs_level(angle)
        if mcs_level is None:
            # 如果仰角超出定义的最高范围，则使用最高档位的MCS等级
            mcs_level = self.MCS_RANGES[-1][2]

        self.mcs_level = mcs_level

        # 根据 MCS 等级直接查找带宽，如果找不到则默认为0
        self.bandwidth = float(self.MCS_BANDWIDTH_MAP.get(mcs_level, 0.0))

    @classmethod
    def _resolve_mcs_level(cls, angle):
        """根据仰角查找对应的MCS等级。"""
        for lower, upper, mcs_level in cls.MCS_RANGES:
            if lower <= angle < upper:
                return mcs_level
        return None

class satellite_beam:
    def __init__(self, user_capacity, capacity) -> None:
        self.connected_user = 0
        self.user_capacity = user_capacity
        self.bandwidth = 0
        self.capacity = capacity
        self.load = []

class gatway:
    def __init__(self, longitude, latitude, capacity, gateway_name=None) -> None:
        self.longitude = longitude
        self.latitude = latitude
        self.gateway_name = gateway_name
        self.capacity = capacity
        self.bandwidth = 0
        self.load = []
        
class central_node:
    """地面中心节点，与信关站构成星型拓扑。"""

    def __init__(self, longitude, latitude, capacity, connected_gateways=None, node_name="中心节点") -> None:
        self.longitude = longitude
        self.latitude = latitude
        self.capacity = capacity
        self.bandwidth = 0
        self.load = []
        self.node_name = node_name
        self.connected_gateways = connected_gateways or []