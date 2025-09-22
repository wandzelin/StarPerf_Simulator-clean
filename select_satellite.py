from utils import distance_between_satellite_and_user

_SELECTION_MODE = "weighted"


def set_selection_mode(mode: str) -> None:
    """Configure satellite selection strategy ('nearest' or 'weighted')."""

    if mode not in {"nearest", "weighted"}:
        raise ValueError(f"Unsupported selection mode: {mode}")
    global _SELECTION_MODE
    _SELECTION_MODE = mode



def select_nearest(connect, U2Svisible_matrix, satellites, t, connect_method):
    # nearest_satellite_to_user = None
    # satellite_to_user_distance = float('inf')
    source_user = connect.source
    target_user = connect.target
    max_bandwidth = connect.max_bandwidth
    min_bandwidth = connect.min_bandwidth
    # need_bandwidth = connect.need_bandwidth
    source_user_row = U2Svisible_matrix[0]
    target_user_row = U2Svisible_matrix[1]

    if connect_method == 't2c' and target_user.bandwidth >= target_user.capacity:
        connect.allocated_bandwidth.append(0)
        return None, None

    source_user_visible_satellite = []
    for edge_item in source_user_row:
        if edge_item.bandwidth != 0 and edge_item.bandwidth >= min_bandwidth:
            source_satellite = edge_item.satellite
            for beam in source_satellite.beams:
                if beam.connected_user < beam.user_capacity and beam.capacity - beam.bandwidth >= min_bandwidth:
                    source_user_visible_satellite.append(edge_item)
                    break

    target_user_visible_satellite = []
    for edge_item in target_user_row:
        if edge_item.bandwidth != 0 and edge_item.bandwidth >= min_bandwidth:
            target_satellite = edge_item.satellite
            for beam in target_satellite.beams:
                if beam.connected_user < beam.user_capacity and beam.capacity - beam.bandwidth >= min_bandwidth:
                    target_user_visible_satellite.append(edge_item)
                    break

    if len(source_user_visible_satellite) == 0 or len(target_user_visible_satellite) == 0:
        connect.allocated_bandwidth.append(0)
        return None, None


    nearest_satellite_to_source = None
    satellite_to_user_distance = float('inf')
    source_bandwidth = 0
    for edge_item in source_user_visible_satellite:
        satellite = edge_item.satellite
        distance = distance_between_satellite_and_user(source_user , satellite , t)
        if distance < satellite_to_user_distance:
            satellite_to_user_distance = distance
            nearest_satellite_to_source = satellite
            source_bandwidth = edge_item.bandwidth

    nearest_satellite_to_target = None
    satellite_to_user_distance = float('inf')
    target_bandwidth = 0
    for edge_item in target_user_visible_satellite:
        satellite = edge_item.satellite
        distance = distance_between_satellite_and_user(target_user , satellite , t)
        if distance < satellite_to_user_distance:
            satellite_to_user_distance = distance
            nearest_satellite_to_target = satellite
            target_bandwidth = edge_item.bandwidth

    if nearest_satellite_to_source == None or nearest_satellite_to_target == None:
        connect.allocated_bandwidth.append(0)
        return None, None

    source_beam = nearest_satellite_to_source.beams[0]
    source_beam_bandwidth = source_beam.bandwidth
    for beam in nearest_satellite_to_source.beams:
        if beam.bandwidth < source_beam_bandwidth:
            source_beam_bandwidth = beam.bandwidth
            source_beam = beam

    target_beam = nearest_satellite_to_target.beams[0]
    target_beam_bandwidth = target_beam.bandwidth
    for beam in nearest_satellite_to_target.beams:
        if beam.bandwidth < target_beam_bandwidth:
            target_beam_bandwidth = beam.bandwidth
            target_beam = beam

    source_beam_can_alloc_bandwidth = min(source_bandwidth, source_beam.capacity - source_beam.bandwidth)
    target_beam_can_alloc_bandwidth = min(target_bandwidth, target_beam.capacity - target_beam.bandwidth)
    if connect_method == 't2t':
        min_beam_can_alloc_bandwidth = min(source_beam_can_alloc_bandwidth, target_beam_can_alloc_bandwidth)
    elif connect_method == 't2c':
        min_beam_can_alloc_bandwidth = min(source_beam_can_alloc_bandwidth, target_beam_can_alloc_bandwidth, target_user.capacity - target_user.bandwidth)

    if min_beam_can_alloc_bandwidth >= max_bandwidth:
        connect.allocated_bandwidth.append(max_bandwidth)
        source_beam.bandwidth += max_bandwidth
        target_beam.bandwidth += max_bandwidth
        source_beam.connected_user += 1
        target_beam.connected_user += 1
        if connect_method == 't2c':
            target_user.bandwidth += max_bandwidth
    else:
        connect.allocated_bandwidth.append(min_beam_can_alloc_bandwidth)
        source_beam.bandwidth += min_beam_can_alloc_bandwidth
        target_beam.bandwidth += min_beam_can_alloc_bandwidth
        source_beam.connected_user += 1
        target_beam.connected_user += 1
        if connect_method == 't2c':
            target_user.bandwidth += min_beam_can_alloc_bandwidth
    return nearest_satellite_to_source, nearest_satellite_to_target


def select_weighted(connect, U2Svisible_matrix, satellites, t, connect_method):
    if _SELECTION_MODE == "nearest":
        return select_nearest(connect, U2Svisible_matrix, satellites, t, connect_method)
    # nearest_satellite_to_user = None
    # satellite_to_user_distance = float('inf')
    source_user = connect.source
    target_user = connect.target
    max_bandwidth = connect.max_bandwidth
    min_bandwidth = connect.min_bandwidth
    # need_bandwidth = connect.need_bandwidth
    source_user_row = U2Svisible_matrix[0]
    target_user_row = U2Svisible_matrix[1]

    if connect_method == 't2c' and target_user.bandwidth >= target_user.capacity:
        connect.allocated_bandwidth.append(0)
        return None, None

    source_user_visible_satellite = []
    for edge_item in source_user_row:
        if edge_item.bandwidth != 0 and edge_item.bandwidth >= min_bandwidth:
            source_satellite = edge_item.satellite
            for beam in source_satellite.beams:
                if beam.connected_user < beam.user_capacity and beam.capacity - beam.bandwidth >= min_bandwidth:
                    source_user_visible_satellite.append(edge_item)
                    break

    target_user_visible_satellite = []
    for edge_item in target_user_row:
        if edge_item.bandwidth != 0 and edge_item.bandwidth >= min_bandwidth:
            target_satellite = edge_item.satellite
            for beam in target_satellite.beams:
                if beam.connected_user < beam.user_capacity and beam.capacity - beam.bandwidth >= min_bandwidth:
                    target_user_visible_satellite.append(edge_item)
                    break

    if len(source_user_visible_satellite) == 0 or len(target_user_visible_satellite) == 0:
        connect.allocated_bandwidth.append(0)
        return None, None


    # ---- 源侧 ----
    import math
    eps = 1e-12

    # 1) 收集距离以及每个候选卫星最空闲可用波束的负载率
    min_d = float('inf')
    max_d = 0.0
    source_dist_list = []
    source_loadratio_list = []
    for edge_item in source_user_visible_satellite:
        sat = edge_item.satellite
        d = distance_between_satellite_and_user(source_user, sat, t)
        source_dist_list.append(d)
        if d < min_d: min_d = d
        if d > max_d: max_d = d
        # 最空闲可用波束的负载率（越小越好）
        best_load_ratio = 1.0
        for beam in sat.beams:
            if beam.capacity > 0:
                load_ratio = beam.bandwidth / beam.capacity
                if load_ratio < best_load_ratio:
                    best_load_ratio = load_ratio
        if best_load_ratio < 0.0:
            best_load_ratio = 0.0
        if best_load_ratio > 1.0:
            best_load_ratio = 1.0
        best_load_ratio = 1.0 - best_load_ratio
        source_loadratio_list.append(best_load_ratio)

    # 2) 距离归一化、并做熵权所需的“正向化”处理（距离列做 y = 1 - x，负载列已转为空闲比）
    src_d_norm_list = []
    for d in source_dist_list:
        if max_d > min_d:
            d_norm = (d - min_d) / (max_d - min_d)
        else:
            d_norm = 0.5
        src_d_norm_list.append(d_norm)

    # 正向化（越大越好，负载列直接使用空闲比）
    y_d_list = [max(0.0, 1.0 - v) for v in src_d_norm_list]
    y_l_list = [max(0.0, v) for v in source_loadratio_list]

    # 3) 熵权计算 α_src（小样本回退 0.5，α 裁剪到 [0.2,0.8]）
    m = len(source_user_visible_satellite)
    sum_y_d = sum(y_d_list)
    sum_y_l = sum(y_l_list)
    if m < 3 or sum_y_d < eps or sum_y_l < eps:
        alpha_src = 0.5
    else:
        p_d = [y / (sum_y_d + eps) for y in y_d_list]
        p_l = [y / (sum_y_l + eps) for y in y_l_list]
        k = 1.0 / (math.log(m) + eps)
        e_d = -k * sum([p * math.log(p + eps) for p in p_d])
        e_l = -k * sum([p * math.log(p + eps) for p in p_l])
        delta_d = 1.0 - e_d
        delta_l = 1.0 - e_l
        denom = delta_d + delta_l
        if denom < eps:
            alpha_src = 0.5
        else:
            alpha_src = delta_d / denom
        # 裁剪
        if alpha_src < 0.2:
            alpha_src = 0.2
        if alpha_src > 0.8:
            alpha_src = 0.8

    # 4) 用 α_src 打分并选择最小 score 的卫星（score = α * distance_norm + (1 - α) * (1 - free_ratio)）
    nearest_satellite_to_source = None
    satellite_to_user_distance = float('inf')  # 这里保存“最小score”
    source_bandwidth = 0
    for idx, edge_item in enumerate(source_user_visible_satellite):
        score = alpha_src * src_d_norm_list[idx] + (1.0 - alpha_src) * (1.0 - source_loadratio_list[idx])
        if score < satellite_to_user_distance:
            satellite_to_user_distance = score
            nearest_satellite_to_source = edge_item.satellite
            source_bandwidth = edge_item.bandwidth

    # ---- 宿侧（同样流程，独立计算 α_dst）----
    min_d = float('inf')
    max_d = 0.0
    target_dist_list = []
    target_loadratio_list = []
    for edge_item in target_user_visible_satellite:
        sat = edge_item.satellite
        d = distance_between_satellite_and_user(target_user, sat, t)
        target_dist_list.append(d)
        if d < min_d: min_d = d
        if d > max_d: max_d = d
        best_load_ratio = 1.0
        for beam in sat.beams:
            if beam.capacity > 0:
                load_ratio = beam.bandwidth / beam.capacity
                if load_ratio < best_load_ratio:
                    best_load_ratio = load_ratio
        if best_load_ratio < 0.0:
            best_load_ratio = 0.0
        if best_load_ratio > 1.0:
            best_load_ratio = 1.0
        best_load_ratio = 1.0 - best_load_ratio
        target_loadratio_list.append(best_load_ratio)

    # 2) 距离归一化、并做熵权所需的“正向化”处理（距离列做 y = 1 - x，负载列已转为空闲比）
    tgt_d_norm_list = []
    for d in target_dist_list:
        if max_d > min_d:
            d_norm = (d - min_d) / (max_d - min_d)
        else:
            d_norm = 0.5
        tgt_d_norm_list.append(d_norm)

    # 正向化（越大越好，负载列直接使用空闲比）
    y_d_list = [max(0.0, 1.0 - v) for v in tgt_d_norm_list]
    y_l_list = [max(0.0, v) for v in target_loadratio_list]

    m = len(target_user_visible_satellite)
    sum_y_d = sum(y_d_list)
    sum_y_l = sum(y_l_list)
    if m < 3 or sum_y_d < eps or sum_y_l < eps:
        alpha_dst = 0.5
    else:
        p_d = [y / (sum_y_d + eps) for y in y_d_list]
        p_l = [y / (sum_y_l + eps) for y in y_l_list]
        k = 1.0 / (math.log(m) + eps)
        e_d = -k * sum([p * math.log(p + eps) for p in p_d])
        e_l = -k * sum([p * math.log(p + eps) for p in p_l])
        delta_d = 1.0 - e_d
        delta_l = 1.0 - e_l
        denom = delta_d + delta_l
        if denom < eps:
            alpha_dst = 0.5
        else:
            alpha_dst = delta_d / denom
        if alpha_dst < 0.2:
            alpha_dst = 0.2
        if alpha_dst > 0.8:
            alpha_dst = 0.8

    nearest_satellite_to_target = None
    satellite_to_user_distance = float('inf')
    target_bandwidth = 0
    for idx, edge_item in enumerate(target_user_visible_satellite):
        score = alpha_dst * tgt_d_norm_list[idx] + (1.0 - alpha_dst) * (1.0 - target_loadratio_list[idx])
        if score < satellite_to_user_distance:
            satellite_to_user_distance = score
            nearest_satellite_to_target = edge_item.satellite
            target_bandwidth = edge_item.bandwidth

    if nearest_satellite_to_source == None or nearest_satellite_to_target == None:
        connect.allocated_bandwidth.append(0)
        return None, None

    source_beam = nearest_satellite_to_source.beams[0]
    source_beam_bandwidth = source_beam.bandwidth
    for beam in nearest_satellite_to_source.beams:
        if beam.bandwidth < source_beam_bandwidth:
            source_beam_bandwidth = beam.bandwidth
            source_beam = beam

    target_beam = nearest_satellite_to_target.beams[0]
    target_beam_bandwidth = target_beam.bandwidth
    for beam in nearest_satellite_to_target.beams:
        if beam.bandwidth < target_beam_bandwidth:
            target_beam_bandwidth = beam.bandwidth
            target_beam = beam

    source_beam_can_alloc_bandwidth = min(source_bandwidth, source_beam.capacity - source_beam.bandwidth)
    target_beam_can_alloc_bandwidth = min(target_bandwidth, target_beam.capacity - target_beam.bandwidth)
    if connect_method == 't2t':
        min_beam_can_alloc_bandwidth = min(source_beam_can_alloc_bandwidth, target_beam_can_alloc_bandwidth)
    elif connect_method == 't2c':
        min_beam_can_alloc_bandwidth = min(source_beam_can_alloc_bandwidth, target_beam_can_alloc_bandwidth, target_user.capacity - target_user.bandwidth)

    if min_beam_can_alloc_bandwidth >= max_bandwidth:
        connect.allocated_bandwidth.append(max_bandwidth)
        source_beam.bandwidth += max_bandwidth
        target_beam.bandwidth += max_bandwidth
        source_beam.connected_user += 1
        target_beam.connected_user += 1
        if connect_method == 't2c':
            target_user.bandwidth += max_bandwidth
    else:
        connect.allocated_bandwidth.append(min_beam_can_alloc_bandwidth)
        source_beam.bandwidth += min_beam_can_alloc_bandwidth
        target_beam.bandwidth += min_beam_can_alloc_bandwidth
        source_beam.connected_user += 1
        target_beam.connected_user += 1
        if connect_method == 't2c':
            target_user.bandwidth += min_beam_can_alloc_bandwidth
    return nearest_satellite_to_source, nearest_satellite_to_target
