# Description: use GHS-POP raster to allocate users onto a 10x10 degree grid
# Notes: ensure at least one user per cell; fallback to uniform weights when raster missing
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from equal_area_partition import EqualAreaCell, EqualAreaGrid, approximate_equal_area_grid


POP_TIF_PATH = "data/pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"  # 人口“人数”GeoTIFF（WGS84, 30″）
TOTAL_USERS = 1944  # 全球总用户数


def _normalise_interval(value: Optional[Sequence[float]], default: Sequence[float]) -> Tuple[float, float]:
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


def _population_weights_from_raster(
    centers: Iterable[Tuple[float, float]],
    grid: EqualAreaGrid,
    tif_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Compute per-cell population weights from raster data."""

    half_lon_deg = math.degrees(grid.step_a / (2.0 * grid.radius_km))
    half_sin = grid.step_b / (2.0 * grid.radius_km)

    if tif_path and os.path.exists(tif_path):
        with rasterio.open(tif_path) as ds:
            weights: List[float] = []
            for lat, lon in centers:
                sin_lat = math.sin(math.radians(lat))
                sin_bottom = max(-1.0, min(1.0, sin_lat - half_sin))
                sin_top = max(-1.0, min(1.0, sin_lat + half_sin))
                bottom = max(lat_min, math.degrees(math.asin(sin_bottom)))
                top = min(lat_max, math.degrees(math.asin(sin_top)))
                left = max(lon_min, lon - half_lon_deg)
                right = min(lon_max, lon + half_lon_deg)
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


def _allocate_uniform(total_users: int, cell_count: int) -> np.ndarray:
    """Deterministically split users evenly across all cells."""

    base = total_users // cell_count
    remainder = total_users % cell_count
    alloc = np.full(cell_count, base, dtype=int)
    if remainder > 0:
        alloc[:remainder] += 1
    return alloc


def _allocate_from_weights(weights: np.ndarray, total_users: int) -> np.ndarray:
    """Convert population weights into integer user counts per cell."""

    normalised = weights / weights.sum()
    raw = normalised * total_users
    floored = np.floor(raw).astype(int)
    remaining = total_users - floored.sum()
    if remaining < 0:
        raise ValueError("整数分配结果超过总用户数，请检查人口权重")
    if remaining > 0:
        residual = raw - floored
        order = np.argsort(-residual)
        take = min(remaining, len(order))
        floored[order[:take]] += 1
        remaining -= take
    if remaining > 0:
        raise ValueError("仍有剩余用户未能分配，请检查权重计算是否正确")
    return floored


def _create_user(USER_module, cell: EqualAreaCell):
    """Instantiate a user at the centre of a grid cell with annotated metadata."""

    user_obj = USER_module.user(cell.longitude, cell.latitude)
    user_obj.cell_id = cell.cell_id
    user_obj.cell_row = cell.i
    user_obj.cell_col = cell.j
    user_obj.cell_latitude = cell.latitude
    user_obj.cell_longitude = cell.longitude
    return user_obj

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
) -> PopulationAllocation:
    """Generate user objects at grid centres based on chosen population distribution."""

    if lat_range is None or lon_range is None:
        raise ValueError("lat_range and lon_range must be provided by the caller")
    if delta_a is None or delta_b is None:
        raise ValueError("delta_a and delta_b must be provided (kilometres)")
    if not distribution:
        raise ValueError("distribution must be specified (e.g. 'density' or 'uniform')")

    distribution = distribution.lower()
    if distribution not in {"uniform", "density"}:
        raise ValueError(f"Unsupported distribution mode: {distribution}")

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

    total_users = int(total_users)
    if total_users < 0:
        raise ValueError("total_users 不能为负数")

    cell_count = grid.m * grid.n
    if distribution == "uniform":
        alloc = _allocate_uniform(total_users, cell_count)
    else:
        weights = _population_weights_from_raster(
            grid.centres,
            grid,
            tif_path,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )
        alloc = _allocate_from_weights(weights, total_users)

    users: List[Any] = []
    cell_user_map: Dict[int, List[Any]] = {cell.cell_id: [] for cell in grid.cells}
    cell_users: List[int] = []

    for cell, count in zip(grid.cells, alloc):
        count = int(count)
        cell_users.append(count)
        for _ in range(count):
            user_obj = _create_user(USER_module, cell)
            users.append(user_obj)
            cell_user_map[cell.cell_id].append(user_obj)

    return PopulationAllocation(
        users=users,
        grid=grid,
        distribution=distribution,
        cell_users=cell_users,
        cell_user_map=cell_user_map,
        total_users=sum(cell_users),
    )
