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
    def __init__(self, user, satellite, t, min_bandwidth, max_bandwidth, min_visible_angle) -> None:
        self.user = user
        self.satellite = satellite
        self.t = t
        self.distance = calculate_distance(user, satellite, t)
        self.bandwidth = None

        sat_x, sat_y, sat_z = latilong_to_descartes(satellite, 'satellite', t)
        user_x, user_y, user_z = latilong_to_descartes(user, 'user', t)
        angle = calculate_angle(sat_x, sat_y, sat_z, user_x, user_y, user_z) - 90
        if angle < min_visible_angle:
            self.bandwidth = 0
        else:
            bw = min_bandwidth + (max_bandwidth - min_bandwidth)/8 * int(8 * angle/(90 - min_visible_angle))
            self.bandwidth = bw
    #123
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