# Description: use GHS-POP raster to allocate users onto a 10x10 degree grid
# Notes: ensure at least one user per cell; fallback to uniform weights when raster missing
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from equal_area_partition import EARTH_RADIUS_KM, approximate_equal_area_grid, EqualAreaGrid


POP_TIF_PATH = "data/pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"  # 人口“人数”GeoTIFF（WGS84, 30″）
TOTAL_USERS  = 1944  # 全球总用户数


def _normalise_interval(value, default):
    if value is None:
        value = default
    start, end = value
    start = float(start)
    end = float(end)
    if start > end:
        start, end = end, start
    if start == end:
        raise ValueError("Interval span must be non-zero.")
    return start, end

def population_distribution_density(
    centers,
    grid,
    tif_path,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    """Compute per-cell population weights from raster data."""

    half_lon_deg = math.degrees(grid.step_a / (2.0 * grid.radius_km))
    half_sin = grid.step_b / (2.0 * grid.radius_km)

    if tif_path and os.path.exists(tif_path):
        with rasterio.open(tif_path) as ds:
            weights = []
            lon_offset = half_lon_deg
            sin_half = half_sin
            for lat, lon in centers:
                sin_lat = math.sin(math.radians(lat))
                sin_bottom = max(-1.0, min(1.0, sin_lat - sin_half))
                sin_top = max(-1.0, min(1.0, sin_lat + sin_half))
                bottom = max(lat_min, math.degrees(math.asin(sin_bottom)))
                top = min(lat_max, math.degrees(math.asin(sin_top)))
                left = max(lon_min, lon - lon_offset)
                right = min(lon_max, lon + lon_offset)
                if right <= left or top <= bottom:
                    weights.append(0.0)
                    continue
                arr = ds.read(
                    1,
                    window=from_bounds(left, bottom, right, top, ds.transform),
                    boundless=True,
                    masked=True,
                ).astype(float)
                if ds.nodata is not None:
                    arr[arr == ds.nodata] = np.nan
                arr[arr < 0] = np.nan  # GHS-POP 常见 NoData = -200
                weights.append(np.nansum(arr))
        w = np.array(weights, dtype=float)
        if not np.isfinite(w).any() or w.sum() <= 0:
            w[:] = 1.0
    else:
        print(f"[INFO] Population raster not found: {tif_path}, fallback to uniform weights")
        w = np.ones(grid.m * grid.n, dtype=float)
    return w

def population_distribution_uniform(cell_count):
    """Return uniform weights when no raster-based weighting is desired."""

    return np.ones(cell_count, dtype=float)

@dataclass
class PopulationAllocation:
    """Container for population allocation results shared across modules."""

    users: List[Any]
    grid: EqualAreaGrid
    distribution: str
    cell_users: List[int]
    cell_user_map: Dict[int, List[Any]]
    total_users: int

def build_users_by_population(
    USER_module,
    total_users,
    tif_path,
    lat_range,
    lon_range,
    delta_a,
    delta_b,
    distribution,
    max_users_per_cell=None,
):
    if lat_range is None or lon_range is None:
        raise ValueError('lat_range and lon_range must be provided by the caller')
    if delta_a is None or delta_b is None:
        raise ValueError('delta_a and delta_b must be provided (kilometres)')
    if not distribution:
        raise ValueError("distribution must be specified (e.g. 'density' or 'uniform')")
    
    """Generate user objects at grid centres based on chosen population distribution."""
    lat_min, lat_max = _normalise_interval(lat_range, lat_range)
    lon_min, lon_max = _normalise_interval(lon_range, lon_range)

    delta_a = float(delta_a) 
    delta_b = float(delta_b) 

    grid = approximate_equal_area_grid(
        (lat_min, lat_max),
        (lon_min, lon_max),
        delta_a,
        delta_b,
    )
    centers = grid.centres
    G = grid.m * grid.n

    total_users = max(total_users, G)  # 保证每格至少 1 用户

    if max_users_per_cell is None:
        max_cap = total_users
    else:
        max_cap = max(1, int(max_users_per_cell))

    # 1) 根据选择的模式获取人口权重
    if distribution == "uniform":
        total_users = G
        alloc = np.ones(G, dtype=int)
    else:
        w = population_distribution_density(
            centers,
            grid,
            tif_path,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )

        raw = w / w.sum() * total_users
        flo = np.floor(raw).astype(int)
        flo = np.minimum(flo, max_cap)
        alloc = flo
        remaining = total_users - alloc.sum()
        if remaining < 0:
            raise ValueError("整数分配结果超过总用户数，请检查人口权重")
        if remaining > 0:
            residual = raw - flo
            available = np.full(G, max_cap, dtype=int) - alloc
            order = np.argsort(-residual)
            for idx in order:
                if remaining <= 0:
                    break
                if available[idx] <= 0:
                    continue
                take = min(available[idx], remaining)
                alloc[idx] += int(take)
                available[idx] -= int(take)
                remaining -= int(take)
            if remaining > 0:
                raise ValueError("仍有剩余用户无法分配到未超上限的小区中")

    # 2) 生成用户对象并记录小区归属
    users: List[Any] = []
    cell_user_map: Dict[int, List[Any]] = {cell.cell_id: [] for cell in grid.cells}
    cell_users = []
    for idx, ((lat, lon), cnt) in enumerate(zip(centers, alloc)):
        cell = grid.cells[idx]
        cell_users.append(int(cnt))
        for _ in range(int(cnt)):
            user_obj = USER_module.user(lat, lon)
            setattr(user_obj, "cell_id", cell.cell_id)
            setattr(user_obj, "cell_row", cell.i)
            setattr(user_obj, "cell_col", cell.j)
            setattr(user_obj, "cell_latitude", cell.latitude)
            setattr(user_obj, "cell_longitude", cell.longitude)
            users.append(user_obj)
            cell_user_map[cell.cell_id].append(user_obj)

    return PopulationAllocation(
        users=users,
        grid=grid,
        distribution=distribution,
        cell_users=cell_users,
        cell_user_map=cell_user_map,
        total_users=int(sum(cell_users)),
    )
